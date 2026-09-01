import pytest
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from pmrf.frequency import Frequency

from scipy.constants import c, mu_0, epsilon_0

from pmrf.models import (
    PhaseLine, 
    RLGCLine, 
    PhysicalLine, 
    DatasheetLine, 
    CoaxialLine, 
    MicrostripLine, 
    StriplineLine,
    FloatingLine
)
from pmrf.materials import (
    BulkConductor,
    ConductorProperties,
    ConstantDielectric,
    DielectricProperties,
    DjordjevicSarkar,
    RoughConductor,
)
from pmrf.materials.conductor_shape import TescheTubeShape
from pmrf.models.components.lines.formulations import (
    TescheCoaxialFormulation,
    KirschningJansenMicrostripDispersion,
)

@pytest.fixture
def basic_freq():
    return Frequency(start=1.0, stop=10.0, npoints=10, unit='GHz')

def test_phase_line(basic_freq):
    """Test ideal, lossless phase line limits."""
    # Create a 50-ohm line, 90 degrees at 5 GHz
    line = PhaseLine(z0=50.0, theta=90.0, f0=5.0e9)
    s = line.s(basic_freq)
    
    assert s.shape == (10, 2, 2)
    
    # S11 should be 0 for a line matched to the system z0 (which defaults to 50)
    assert jnp.allclose(s[:, 0, 0], 0.0, atol=1e-5)
    
    # Magnitude of transmission should be perfectly 1.0 (lossless)
    assert jnp.allclose(jnp.abs(s[:, 1, 0]), 1.0, atol=1e-5)

def test_floating_line(basic_freq):
    """Test the wrapper that converts a 2-port to a 4-port floating line."""
    base_line = PhaseLine(z0=50.0, theta=90.0, f0=5e9)
    float_line = FloatingLine(floating=base_line)
    
    s = float_line.s(basic_freq)
    
    # Should automatically scale up to a 4-port parameter matrix
    assert s.shape == (10, 4, 4)
    assert not jnp.any(jnp.isnan(s))

def test_constant_rlgc_line(basic_freq):
    """Test a basic transmission line evaluated from per-unit-length components."""
    # Lossless 50 ohm line (L=250nH, C=100pF -> Zc = sqrt(L/C) = 50)
    line = RLGCLine(R=0.0, G=0.0, L=250e-9, C=100e-12, length=0.1)
    s = line.s(basic_freq)
    
    assert s.shape == (10, 2, 2)
    assert jnp.allclose(s[:, 0, 0], 0.0, atol=1e-5)
    assert jnp.allclose(jnp.abs(s[:, 1, 0]), 1.0, atol=1e-5)

@pytest.mark.parametrize("line_model", [
    PhysicalLine(length=0.1),
    DatasheetLine(length=0.1),
    CoaxialLine(length=0.1),
    MicrostripLine(length=0.1)
])
def test_physical_line_execution(line_model, basic_freq):
    """
    Ensure all physical models compile and evaluate correctly without 
    divide-by-zero or shape mismatch errors.
    """
    s = line_model.s(basic_freq)
    assert s.shape == (10, 2, 2)
    assert not jnp.any(jnp.isnan(s))


def test_physical_line_phase_delay(basic_freq):
    """Test that PhysicalLine calculates the correct phase delay based on permittivity."""
    # 15 mm line in a dielectric with Er = 4.0
    length = 0.015
    ep_r = 4.0
    line = PhysicalLine(zn=50.0, length=length, ep_r=ep_r, A=0.0, tand=0.0)
    
    s = line.s(basic_freq)
    
    # Expected phase shift: beta * L = w * sqrt(ep_r) * L / c
    # Since S21 = exp(-j * beta * L), angle(S21) = -beta * L
    expected_phase = -basic_freq.w * jnp.sqrt(ep_r) * length / c
    actual_phase = jnp.unwrap(jnp.angle(s[:, 1, 0]))
    
    assert jnp.allclose(actual_phase, expected_phase, atol=1e-4)
    # Lossless matched line -> S11 should be exactly 0
    assert jnp.allclose(s[:, 0, 0], 0.0, atol=1e-5)

def test_datasheet_line_attenuation(basic_freq):
    """Test that the DatasheetLine applies the correct frequency-scaled attenuation."""
    # Pure skin-effect loss (k1), no dielectric loss (k2), normalized coefficients
    line = DatasheetLine(
        zn=50.0, vf=1.0, k1=2.0, k2=0.0, length=10.0, loss_coeffs_normalized=True
    )
    
    s = line.s(basic_freq)
    
    # alpha_c = k1_norm * ln(10)/20 * sqrt(w)
    # |S21| = exp(-alpha_c * L)
    alpha_c = 2.0 * (jnp.log(10) / 20) * jnp.sqrt(basic_freq.w)
    expected_mag = jnp.exp(-alpha_c * 10.0)
    
    assert jnp.allclose(jnp.abs(s[:, 1, 0]), expected_mag, atol=1e-4)

def test_coaxial_line_impedance(basic_freq):
    """Verify CoaxialLine matches the analytical Zc formula for a lossless coax."""
    # RG-58 dimensions roughly: ep_r=2.25, d_in=0.9mm, d_out=2.95mm
    line = CoaxialLine(
        d_in=0.9e-3,
        d_out=2.95e-3,
        dielectric=ConstantDielectric(ep_r=2.25, tand=0.0),
        conductor=BulkConductor(sigma=jnp.inf),
        length=1.0,
    )
    
    # Extract Zc directly from the internal method
    zc, _ = line.zc_and_gammaL(basic_freq)
    
    # Analytical Zc = (1 / 2*pi) * sqrt(mu / eps) * ln(b/a)
    # (Note: 1/(2*pi) * sqrt(mu_0/eps_0) is approx 59.958 ohms)
    expected_zc = (1 / (2 * jnp.pi)) * jnp.sqrt(mu_0 / (epsilon_0 * 2.25)) * jnp.log(2.95 / 0.9)
    
    # Real part of Zc should match the theoretical lossless impedance
    assert jnp.allclose(jnp.real(zc), expected_zc, rtol=1e-3)


def test_coaxial_static_conductivity_has_analytic_dc_conductance():
    freq = Frequency.from_f(jnp.array([0.0, 1.0, 1e3]))
    sigma = 0.01
    d_in, d_out = 0.9e-3, 2.95e-3
    line = CoaxialLine(
        d_in=d_in,
        d_out=d_out,
        dielectric=ConstantDielectric(ep_r=2.25, sigma=sigma),
        conductor=BulkConductor(sigma=jnp.inf),
        length=1.0,
    )

    expected = 2 * jnp.pi * sigma / jnp.log(d_out / d_in)
    assert jnp.allclose(line.immittance(freq).G, expected)


def test_coaxial_rejects_reversed_diameters():
    line = CoaxialLine(d_in=2e-3, d_out=1e-3, length=0.1)
    with pytest.raises(Exception, match="d_out must exceed d_in"):
        line.immittance(Frequency.from_f(jnp.array([1e9])))

def test_coaxial_permeability_comes_from_the_dielectric(basic_freq):
    """A magnetic filling is a property of the material, not of the geometry.

    Zc = sqrt(mu/eps) * ln(b/a) / 2pi, so mu_r = 4 raises it by exactly 2.
    """
    def coax(dielectric):
        return CoaxialLine(
            d_in=0.9e-3,
            d_out=2.95e-3,
            dielectric=dielectric,
            conductor=BulkConductor(sigma=jnp.inf),
            length=1.0,
        )

    plain = coax(ConstantDielectric(ep_r=2.25, tand=0.0))
    magnetic = coax(ConstantDielectric(ep_r=2.25, tand=0.0, mu_r=4.0))

    zc_plain, _ = plain.zc_and_gammaL(basic_freq)
    zc_magnetic, _ = magnetic.zc_and_gammaL(basic_freq)

    assert jnp.allclose(jnp.real(zc_magnetic), 2.0 * jnp.real(zc_plain), rtol=1e-6)


def test_coaxial_magnetic_loss_enters_the_series_resistance(basic_freq):
    """A complex mu_r is not restricted at the interface; it becomes loss.

    With a lossless conductor and a lossless dielectric the only remaining loss
    channel is magnetic: Im(mu_r) < 0 adds w*mu''*ln(b/a)/2pi to Re(Z).
    """
    class LossyMagnetic(ConstantDielectric):
        def properties(self, freq):
            properties = super().properties(freq)
            return eqx.tree_at(
                lambda p: p.mu_r,
                properties,
                (4.0 - 0.5j) * jnp.ones(freq.npoints, dtype=complex),
            )

    line = CoaxialLine(
        d_in=0.9e-3,
        d_out=2.95e-3,
        dielectric=LossyMagnetic(ep_r=2.25, tand=0.0),
        conductor=BulkConductor(sigma=jnp.inf),
        length=1.0,
    )
    immittance = line.immittance(basic_freq)

    expected_R = (
        basic_freq.w * 0.5 * mu_0 * jnp.log(2.95 / 0.9) / (2 * jnp.pi)
    )
    assert jnp.allclose(immittance.R, expected_R, rtol=1e-6)
    # A passive magnetic loss must not show up as a negative resistance.
    assert jnp.all(immittance.R > 0)


def test_microstrip_line_default_construction_has_conductor_loss(basic_freq):
    """A default-constructed line has nonzero, sigma-sensitive attenuation.

    Regression for the bug where Wheeler's conductor-loss correction was
    guarded on `substrate.t is not None`, and `t` defaults to `None`: a
    default-constructed line (dispersion=KirschningJansenMicrostripDispersion,
    t=None) had `sigma` completely inert.
    """
    def alpha(sigma):
        line = MicrostripLine(
            dielectric=ConstantDielectric(ep_r=4.3, tand=0.0),
            conductor=BulkConductor(sigma=sigma),
            length=0.1,
        )
        _, gamma_length = line.zc_and_gammaL(basic_freq)
        return jnp.real(gamma_length / line.length)

    alpha_lossless = alpha(jnp.inf)
    alpha_lossy = alpha(1 / 1.68e-8)

    assert jnp.all(alpha_lossless == 0.0)
    assert jnp.all(alpha_lossy > alpha_lossless)


def test_microstrip_line_impedance(basic_freq):
    """Verify MicrostripLine evaluates Wheeler's formula correctly."""
    # Standard 50-ohm trace on 1.6mm FR4 (Er=4.3) is roughly 3.0mm wide
    line = MicrostripLine(
        w=3.0e-3,
        h=1.6e-3,
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.0),
        conductor=BulkConductor(sigma=jnp.inf),
        dispersion=None,
        length=0.1,
    )
    
    zc, _ = line.zc_and_gammaL(basic_freq)
    
    # The Wheeler approximation for this geometry should yield an impedance near 50 ohms
    zc_real = jnp.real(zc)
    assert jnp.all(zc_real > 48.0)
    assert jnp.all(zc_real < 52.0)

@pytest.mark.parametrize("dispersion", [None, KirschningJansenMicrostripDispersion()])
def test_microstrip_line_ep_eff_and_w_eff_match_immittance_pipeline(basic_freq, dispersion):
    """`ep_eff`/`w_eff` must report exactly what `immittance` uses internally.

    Regression for #80: these were previously reconstructed by hand in the
    scikit-rf validation matrix rather than exposed on the model, so a
    refactor of `immittance` could silently drift from the recomputed value.
    Recomputes the quasi-static (+ optional dispersion) pipeline independently
    here, rather than calling the model's own helper, so the test does not
    just check the method against itself.
    """
    w, h, ep_r = 3.0e-3, 1.6e-3, 4.3
    line = MicrostripLine(
        w=w,
        h=h,
        dielectric=ConstantDielectric(ep_r=ep_r, tand=0.01),
        conductor=BulkConductor(sigma=1 / 1.68e-8),
        dispersion=dispersion,
        length=0.1,
    )

    dielectric = line.substrate.dielectric.properties(basic_freq)
    quasi_static = line.formulation.quasi_static(w=w, h=h, t=None, ep_r=dielectric.ep_r)
    if dispersion is None:
        expected_ep_eff, expected_w_eff = quasi_static.ep_eff, quasi_static.w_eff
    else:
        expected_ep_eff, _ = dispersion.disperse(
            basic_freq, ep_eff_0=quasi_static.ep_eff, zc_0=quasi_static.zc,
            ep_r=dielectric.ep_r, w_eff=quasi_static.w_eff, h=h,
        )
        expected_w_eff = quasi_static.w_eff

    assert jnp.allclose(line.ep_eff(basic_freq), expected_ep_eff)
    assert jnp.allclose(line.w_eff(basic_freq), expected_w_eff)
    assert jnp.iscomplexobj(line.ep_eff(basic_freq))
    assert jnp.any(jnp.imag(line.ep_eff(basic_freq)) != 0.0)


def test_microstrip_finite_thickness_has_dc_resistance_floor():
    """#84: a finite-t line's R settles at rho/(W*t) at dc instead of 0.

    The sheet model alone gives R -> 0 as f -> 0, wrong for a trace of known
    thickness. A finite t gets a floor; t=None gets none, since it asserts
    skin effect in operation at every frequency including dc, leaving no dc
    regime for a floor to describe (matches ADS, which floors nothing).
    """
    from pmrf.models import HammerstadJensenMicrostripFormulation

    w, h, t, rho = 3.0e-3, 1.6e-3, 35e-6, 1.68e-8
    dc = Frequency.from_f(jnp.array([0.0]))

    floored = MicrostripLine(
        w=w, h=h, t=t,
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.0),
        conductor=BulkConductor(sigma=1 / rho),
        formulation=HammerstadJensenMicrostripFormulation(),
        length=0.1,
    )
    r_dc_expected = rho / (w * t)
    assert jnp.allclose(floored.immittance(dc).R, r_dc_expected)

    unfloored = MicrostripLine(
        w=w, h=h, t=None,
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.0),
        conductor=BulkConductor(sigma=1 / rho),
        length=0.1,
    )
    assert jnp.allclose(unfloored.immittance(dc).R, 0.0)


def test_material_coercion():
    """Scalars and tuples coerce into the corresponding material modules."""
    line = CoaxialLine(dielectric=(2.25, 0.001), conductor=5.8e7, length=0.1)

    assert isinstance(line.dielectric, ConstantDielectric)
    assert isinstance(line.conductor, BulkConductor)
    assert jnp.allclose(line.dielectric.ep_r.value, 2.25)


def test_coaxial_formulation_takes_plain_arrays(basic_freq):
    """A coaxial formulation is callable without ParamRF objects."""
    npoints = basic_freq.npoints
    result = TescheCoaxialFormulation().immittance(
        basic_freq,
        d_in=0.9e-3,
        d_out=2.95e-3,
        dielectric=DielectricProperties(
            np.full(npoints, 2.25 - 0.00225j), np.ones(npoints), np.zeros(npoints)
        ),
        conductor=ConductorProperties(
            np.full(npoints, 0.01 + 0.01j),
            np.full(npoints, 1 / 1.72e-8),
            np.ones(npoints),
        ),
    )
    assert result.Z.shape == (npoints,)
    assert result.Y.shape == (npoints,)


def test_coaxial_defaults_reuse_inner_material_and_infinite_wall(basic_freq):
    """Unspecified shield options preserve the simple coaxial construction."""
    line = CoaxialLine(length=0.1)
    explicit = CoaxialLine(
        length=0.1, outer_conductor=line.conductor, shield_thickness=None
    )
    assert jnp.allclose(line.immittance(basic_freq).Z, explicit.immittance(basic_freq).Z)


def test_coaxial_supports_dissimilar_finite_shield(basic_freq):
    """The shield can be a separate metal and a finite wall changes loss."""
    inner = BulkConductor(sigma=5.8e7)
    outer = BulkConductor(sigma=3.5e7)
    infinite = CoaxialLine(
        conductor=inner, outer_conductor=outer, length=0.1
    )
    finite = CoaxialLine(
        conductor=inner, outer_conductor=outer, shield_thickness=12.5e-6,
        length=0.1,
    )
    assert jnp.all(finite.immittance(basic_freq).R > 0)
    assert not jnp.allclose(
        finite.immittance(basic_freq).R, infinite.immittance(basic_freq).R
    )


def test_coaxial_internal_impedance_tube():
    """Tube arithmetic follows Tesche's equation (13).

    Kept here rather than folded into the wideband scikit-rf matrix in
    ``test_lines_skrf_matrix.py``: it pins one formula on its own, so a failure
    points straight at the internal impedance instead of at whichever line
    happens to use it.
    """
    skrf = pytest.importorskip("skrf")
    from skrf.media import Coaxial

    freq = Frequency.from_f(jnp.array([1e9]))
    sigma, mu = jnp.array([1e7]), jnp.ones(1)
    zs = jnp.sqrt(1j * freq.w * mu_0 / sigma)
    radius, thickness = 1e-3, 0.2e-3
    reference = Coaxial(
        freq.to_skrf(), Dint=2 * radius, Dout=4 * radius,
        sigma=float(sigma[0]), tout=thickness, model="tesche",
    )
    expected = reference._conductor_impedance(radius, thickness, None)
    conductor = ConductorProperties(zs, sigma, mu)
    zs_sq = TescheTubeShape().impedance(freq.w, conductor, a=radius, t=thickness)
    actual = zs_sq / (2 * jnp.pi * radius)
    assert jnp.allclose(actual, expected)


def test_rough_conductor_is_passed_to_coaxial_formulation():
    freq = Frequency.from_f(jnp.array([1e6, 1e9]))
    smooth = CoaxialLine(conductor=BulkConductor(1 / 1.68e-8), length=0.1)
    rough = CoaxialLine(
        conductor=RoughConductor(1 / 1.68e-8, roughness=1e-6), length=0.1
    )
    assert jnp.all(rough.immittance(freq).R > smooth.immittance(freq).R)


def test_immittance_rlgc_roundtrip(basic_freq):
    """R, L, G, C are exact derived views on Z and Y."""
    line = RLGCLine(R=0.5, G=1e-4, L=250e-9, C=100e-12, length=0.1)
    imm = line.immittance(basic_freq)

    assert jnp.allclose(imm.R, 0.5)
    assert jnp.allclose(imm.L, 250e-9)
    assert jnp.allclose(imm.G, 1e-4)
    assert jnp.allclose(imm.C, 100e-12)


def test_immittance_rlgc_at_dc():
    """L and C stay finite when the axis includes DC."""
    freq = Frequency(start=0.0, stop=1.0, npoints=3, unit='GHz')
    line = RLGCLine(R=0.5, G=1e-4, L=250e-9, C=100e-12, length=0.1)
    imm = line.immittance(freq)

    assert jnp.all(jnp.isfinite(imm.L))
    assert jnp.all(jnp.isfinite(imm.C))
    assert jnp.allclose(imm.L, 250e-9)
    assert jnp.allclose(imm.C, 100e-12)


def test_immittance_inversion_rejects_non_passive_result():
    """Exact (Zc, gamma) inversion surfaces non-physical empirical output."""
    from pmrf.models import ImmittanceResult

    with pytest.raises(eqx.EquinoxRuntimeError, match=r"Re\(Z\) < 0"):
        ImmittanceResult.from_zc_gamma(
            zc=jnp.array([50.0]), gamma=jnp.array([-1.0 + 1j]), w=jnp.array([1.0])
        )


def test_dispersive_dielectric_is_grid_independent():
    """
    A dispersive material is a function of frequency alone, so the same line
    evaluated on two different grids agrees at the frequencies they share.
    """
    line = CoaxialLine(
        d_in=0.9e-3,
        d_out=2.95e-3,
        dielectric=DjordjevicSarkar(ep_r=4.3, tand=0.02),
        conductor=BulkConductor(sigma=1 / 1.68e-8),
        length=0.1,
    )

    coarse = Frequency(start=1.0, stop=5.0, npoints=5, unit='GHz')
    fine = Frequency(start=1.0, stop=9.0, npoints=9, unit='GHz')

    s_coarse = line.s(coarse)
    s_fine = line.s(fine)

    # The coarse grid is 1, 2, 3, 4, 5 GHz; the fine grid's first five points.
    assert jnp.allclose(s_coarse, s_fine[:5], atol=1e-12)


def test_microstrip_formulation_takes_plain_arrays(basic_freq):
    """A formulation is pure numerics, callable with no ParamRF objects."""
    from pmrf.models import WheelerMicrostripFormulation

    npoints = basic_freq.npoints
    ep_r = np.full(npoints, 4.3 - 0.086j)
    zs = np.full(npoints, 0.01 + 0.01j)

    result = WheelerMicrostripFormulation().quasi_static(
        w=3.0e-3, h=1.6e-3, t=None, ep_r=ep_r
    )

    assert result.ep_eff.shape == (npoints,)
    assert jnp.all(jnp.real(result.zc) > 40.0)
    assert jnp.allclose(result.w_eff, 3.0e-3)


def test_dispersive_microstrip_routes_through_planar_quasi_static_to_immittance():
    """#83: the dispersion path reports true RLGC Zc, not modal K-J Zc.

    Inverting the dispersed modal ``(zc, gamma)`` through ``from_zc_gamma``
    made ``Zc = sqrt(Z/Y)`` tautologically reproduce Kirschning-Jansen's own
    modal Zc, so the conductor never genuinely entered it. The dispersion
    path must instead build a fresh
    :class:`~pmrf.models.components.lines.formulations.PlanarQuasiStaticResult`
    at the dispersed ``(ep_eff, zc)`` and route it through
    :meth:`~pmrf.models.components.lines.formulations.PlanarQuasiStaticResult.to_immittance`,
    exactly like the quasi-static path, so ``line.immittance`` matches that
    construction bit for bit.
    """
    from pmrf.models import HammerstadJensenMicrostripFormulation, KirschningJansenMicrostripDispersion
    from pmrf.models.components.lines.formulations import (
        PlanarQuasiStaticResult,
        _wheeler_conductor_loss_factor,
    )

    freq = Frequency(start=1.0, stop=50.0, npoints=51, unit="GHz")
    formulation = HammerstadJensenMicrostripFormulation()
    dispersion = KirschningJansenMicrostripDispersion()
    line = MicrostripLine(
        w=0.2e-3,
        h=0.5e-3,
        t=18e-6,
        dielectric=ConstantDielectric(ep_r=10.0, tand=0.05),
        conductor=BulkConductor(sigma=1e6),
        formulation=formulation,
        dispersion=dispersion,
        length=0.1,
    )

    dielectric = line.substrate.dielectric.properties(freq)
    conductor = line.substrate.conductor.properties(freq)
    quasi_static = formulation.quasi_static(
        w=line.w, h=line.substrate.h, t=line.substrate.t, ep_r=dielectric.ep_r
    )
    ep_eff, zc = dispersion.disperse(
        freq,
        ep_eff_0=quasi_static.ep_eff,
        zc_0=quasi_static.zc,
        ep_r=dielectric.ep_r,
        w_eff=quasi_static.w_eff,
        h=line.substrate.h,
    )
    expected = PlanarQuasiStaticResult(
        ep_eff=ep_eff,
        zc=zc,
        w_eff=quasi_static.w_eff,
        shunt_conductance_factor=quasi_static.shunt_conductance_factor,
    ).to_immittance(
        freq, dielectric, conductor,
        current_distribution=line.current_distribution,
        w=line.w, h=line.substrate.h, t=line.substrate.t,
    )

    actual = line.immittance(freq)
    assert jnp.array_equal(actual.Z, expected.Z)
    assert jnp.array_equal(actual.Y, expected.Y)

    # The dispersed zc no longer equals sqrt(Z/Y) tautologically: the
    # conductor genuinely enters the exact RLGC ratio, so the two differ.
    exact_zc = jnp.sqrt(actual.Z / actual.Y)
    assert not jnp.allclose(exact_zc, zc, rtol=1e-6, atol=1e-6)


def test_microstrip_without_dispersion_preserves_quasi_static_immittance():
    """Disabling stage three reproduces the pre-dispersion pipeline exactly."""
    freq = Frequency(start=1.0, stop=20.0, npoints=21, unit="GHz")
    line = MicrostripLine(
        w=3e-3,
        h=1.6e-3,
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
        conductor=BulkConductor(sigma=1 / 1.72e-8),
        dispersion=None,
        length=0.1,
    )

    ep_r = line.substrate.dielectric.properties(freq).ep_r
    zs = line.substrate.conductor.properties(freq).zs
    quasi_static = line.formulation.quasi_static(
        w=line.w, h=line.substrate.h, t=line.substrate.t, ep_r=ep_r
    )

    actual = line.immittance(freq)
    expected = quasi_static.to_immittance(
        freq,
        line.substrate.dielectric.properties(freq),
        line.substrate.conductor.properties(freq),
        current_distribution=line.current_distribution,
        w=line.w,
        h=line.substrate.h,
        t=line.substrate.t,
    )
    assert jnp.array_equal(actual.Z, expected.Z)
    assert jnp.array_equal(actual.Y, expected.Y)


def test_microstrip_dispersion_is_a_pure_dispersion_toggle():
    """Turning dispersion off must not change *which* conductor loss applies.

    Both the quasi-static path (``PlanarQuasiStaticResult.to_immittance``) and
    the dispersion path charge the selected trace/ground strategy, so
    disabling dispersion with an identity dispersion formulation (one that
    hands the quasi-static ep_eff/zc straight through, unchanged) reproduces
    the quasi-static attenuation to the accuracy of the quasi-static path's
    own low-loss linearisation -- not to the ~9% gap that opened when the two
    paths charged different conductor-loss forms (issue #82).
    """
    from pmrf.models import HammerstadJensenMicrostripFormulation
    from pmrf.models.components.lines.formulations import AbstractMicrostripDispersion

    class IdentityDispersion(AbstractMicrostripDispersion):
        """Hands the quasi-static ep_eff/zc through unchanged."""

        def disperse(self, freq, *, ep_eff_0, zc_0, ep_r, w_eff, h):
            return ep_eff_0, zc_0

    freq = Frequency(start=1.0, stop=20.0, npoints=21, unit="GHz")

    def alpha(dispersion):
        line = MicrostripLine(
            w=3e-3,
            h=1.6e-3,
            t=35e-6,
            dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
            conductor=BulkConductor(sigma=1 / 1.72e-8),
            formulation=HammerstadJensenMicrostripFormulation(),
            dispersion=dispersion,
            length=0.1,
        )
        _, gamma_length = line.zc_and_gammaL(freq)
        return jnp.real(gamma_length / line.length)

    alpha_quasi_static = alpha(None)
    alpha_identity_dispersion = alpha(IdentityDispersion())

    assert jnp.allclose(alpha_quasi_static, alpha_identity_dispersion, rtol=1e-2)


def test_microstrip_defaults_to_kirschning_jansen():
    """Modal dispersion is the accuracy-oriented microstrip default."""
    from pmrf.models import KirschningJansenMicrostripDispersion

    assert isinstance(MicrostripLine(length=0.1).dispersion, KirschningJansenMicrostripDispersion)


def test_hammerstad_jensen_finite_thickness_matches_skrf():
    """The thickness-aware quasi-static formulation agrees with scikit-rf.

    Kept as a local unit test for the same reason as
    ``test_coaxial_internal_impedance_tube``: it exercises the quasi-static
    solution alone, with no dispersion, loss or conversion on top of it. The
    wideband end-to-end comparisons live in ``test_lines_skrf_matrix.py``.
    """
    from skrf.media import MLine
    from pmrf.models import HammerstadJensenMicrostripFormulation

    freq = Frequency(start=1.0, stop=10.0, npoints=10, unit="GHz")
    formulation = HammerstadJensenMicrostripFormulation()
    ep_r = jnp.full(freq.npoints, 4.3)
    result = formulation.quasi_static(w=1e-3, h=1e-3, t=35e-6, ep_r=ep_r)
    media = MLine(
        freq.to_skrf(),
        w=1e-3,
        h=1e-3,
        t=35e-6,
        ep_r=4.3,
        tand=0.0,
        rho=1.68e-8,
        rough=0.0,
        model="hammerstadjensen",
        disp="none",
        diel="frequencyinvariant",
    )

    assert jnp.allclose(result.zc, media.z0_characteristic, rtol=1e-12)
    assert jnp.allclose(result.ep_eff, media.ep_reff, rtol=1e-12)
    assert jnp.allclose(result.w_eff, media.w_eff, rtol=1e-12)


def test_microstrip_near_air_has_finite_conductance_and_gradient():
    """The old filling-factor singularity cannot return at εr approaching one."""
    freq = Frequency(start=1.0, stop=5.0, npoints=5, unit="GHz")

    def response(ep_r):
        line = MicrostripLine(
            w=1e-3,
            h=1e-3,
            dielectric=ConstantDielectric(ep_r=ep_r, tand=0.02),
            conductor=BulkConductor(sigma=jnp.inf),
            length=0.1,
        )
        return jnp.real(line.s(freq)[-1, 0, 0])

    near_air = MicrostripLine(
        w=1e-3,
        h=1e-3,
        dielectric=ConstantDielectric(ep_r=1.0 + 1e-12, tand=0.02),
        conductor=BulkConductor(sigma=jnp.inf),
        length=0.1,
    )
    near_air_g = near_air.immittance(freq).G
    slightly_above = MicrostripLine(
        w=1e-3,
        h=1e-3,
        dielectric=ConstantDielectric(ep_r=1.0 + 2e-12, tand=0.02),
        conductor=BulkConductor(sigma=jnp.inf),
        length=0.1,
    ).immittance(freq).G
    assert jnp.all(jnp.isfinite(near_air_g))
    assert jnp.allclose(near_air_g, slightly_above, rtol=1e-10, atol=1e-12)

    gradients = jax.vmap(jax.grad(response))(jnp.linspace(1.0 + 1e-12, 12.0, 25))
    assert jnp.all(jnp.isfinite(gradients))


def test_coaxial_line_has_no_modal_dispersion_field():
    """Homogeneously filled coax has no modal-dispersion stage."""
    assert not hasattr(CoaxialLine(length=0.1), "dispersion")


# -----------------------------------------------------------------------------
# Stripline
#
# scikit-rf has no stripline media, so these compare against the published
# closed-form results (Pozar, Microwave Engineering 4th ed., Section 3.7)
# and against the exact identities a homogeneously-filled line must satisfy.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("line", [
    MicrostripLine(
        dielectric=ConstantDielectric(ep_r=4.3, sigma=0.01), length=0.1
    ),
    StriplineLine(
        dielectric=ConstantDielectric(ep_r=2.2, sigma=0.01), length=0.1
    ),
])
def test_planar_static_conductivity_is_finite_and_continuous_at_dc(line):
    freq = Frequency.from_f(jnp.array([0.0, 1e-3, 1.0]))
    conductance = line.immittance(freq).G
    assert jnp.all(jnp.isfinite(conductance))
    assert jnp.all(conductance > 0)
    assert jnp.allclose(conductance[0], conductance[1], rtol=1e-12)


def test_microstrip_rejects_magnetic_substrate():
    line = MicrostripLine(
        dielectric=ConstantDielectric(ep_r=4.3, mu_r=2.0), length=0.1
    )
    with pytest.raises(Exception, match="nonmagnetic substrate"):
        line.immittance(Frequency.from_f(jnp.array([1e9])))


def test_stripline_rejects_thickness_not_below_ground_spacing():
    line = StriplineLine(b=1e-3, t=1e-3, length=0.1)
    with pytest.raises(Exception, match="0 < t < b"):
        line.immittance(Frequency.from_f(jnp.array([1e9])))

def test_stripline_effective_permittivity_equals_relative_permittivity():
    """Stripline is homogeneously filled, so there is no filling factor at all."""
    from pmrf.models import CohnStriplineFormulation, StriplineLine

    freq = Frequency(start=1.0, stop=20.0, npoints=11, unit="GHz")
    dielectric = ConstantDielectric(ep_r=2.2, tand=0.001)
    line = StriplineLine(w=2.655e-3, b=3.2e-3, dielectric=dielectric, length=0.1)

    quasi_static = CohnStriplineFormulation().quasi_static(
        w=line.w,
        b=line.b,
        t=line.t,
        ep_r=dielectric.properties(freq).ep_r,
    )

    assert jnp.array_equal(quasi_static.ep_eff, dielectric.properties(freq).ep_r)


def test_stripline_impedance_matches_pozar_design_example():
    """Pozar Example 3.7: er=2.20, b=0.32 cm, W/b=0.830 designs a 50 ohm line."""
    from pmrf.models import StriplineLine

    freq = Frequency(start=10.0, stop=10.0, npoints=1, unit="GHz")
    b = 0.32e-2
    w_over_b = 30 * jnp.pi / (jnp.sqrt(2.2) * 50.0) - 0.441
    line = StriplineLine(
        w=w_over_b * b,
        b=b,
        dielectric=2.2,
        conductor=BulkConductor(sigma=jnp.inf),
        length=0.1,
    )

    zc = jnp.real(line.zc(freq))
    assert jnp.allclose(zc, 50.0, rtol=2e-3)


def test_stripline_dielectric_loss_is_the_homogeneous_limit():
    """A homogeneous line has alpha_d = k tan(delta) / 2 exactly."""
    from pmrf.models import StriplineLine

    freq = Frequency(start=10.0, stop=10.0, npoints=1, unit="GHz")
    tand = 0.001
    line = StriplineLine(
        w=2.655e-3,
        b=3.2e-3,
        dielectric=ConstantDielectric(ep_r=2.2, tand=tand),
        conductor=BulkConductor(sigma=jnp.inf),
        length=1.0,
    )

    alpha = jnp.real(line.gammaL(freq))
    k = freq.w * jnp.sqrt(2.2) / c
    assert jnp.allclose(alpha, k * tand / 2, rtol=1e-3)


def test_stripline_attenuation_matches_pozar_worked_example():
    """Pozar Example 3.7 publishes alpha_d = 0.155 Np/m and alpha_c = 0.122 Np/m.

    Geometry: er=2.20, tand=0.001, b=0.32 cm, t=0.01 mm, copper, Z0=50 ohm,
    at 10 GHz.
    """
    from pmrf.models import StriplineLine

    freq = Frequency(start=10.0, stop=10.0, npoints=1, unit="GHz")
    geometry = dict(w=2.655e-3, b=3.2e-3, t=1e-5, length=1.0)
    copper = BulkConductor(sigma=1 / 1.72e-8)

    lossless = StriplineLine(
        dielectric=2.2, conductor=BulkConductor(sigma=jnp.inf), **geometry
    )
    conductor_only = StriplineLine(dielectric=2.2, conductor=copper, **geometry)
    dielectric_only = StriplineLine(
        dielectric=ConstantDielectric(ep_r=2.2, tand=0.001),
        conductor=BulkConductor(sigma=jnp.inf),
        **geometry,
    )

    assert jnp.allclose(jnp.real(lossless.zc(freq)), 50.0, rtol=1e-3)
    assert jnp.allclose(jnp.real(lossless.gammaL(freq)), 0.0, atol=1e-12)
    # Pozar rounds to three digits, so match to that.
    assert jnp.allclose(jnp.real(conductor_only.gammaL(freq)), 0.122, atol=5e-4)
    assert jnp.allclose(jnp.real(dielectric_only.gammaL(freq)), 0.155, atol=5e-4)


def test_stripline_conductor_loss_scales_with_root_frequency():
    """Conductor loss enters through the surface impedance, so it follows sqrt(f)."""
    from pmrf.models import StriplineLine

    freq = Frequency(start=1.0, stop=100.0, npoints=3, unit="GHz")
    line = StriplineLine(
        w=2.655e-3,
        b=3.2e-3,
        t=35e-6,
        dielectric=2.2,
        conductor=BulkConductor(sigma=1 / 1.72e-8),
        length=1.0,
    )

    alpha = jnp.real(line.gammaL(freq))
    assert jnp.all(alpha > 0)
    # A hundredfold in frequency is a tenfold in skin-effect attenuation.
    assert jnp.allclose(alpha[2] / alpha[0], 10.0, rtol=1e-2)


@pytest.mark.parametrize(
    (
        "w", "b", "t", "expected_transition_r", "transition_rtol",
        "expected_transition_x", "transition_x_rtol", "expected_skin_weight",
    ),
    [
        (
            2.655e-3, 3.2e-3, 35e-6,
            0.2766344102, 2e-8, 0.0129251261, 2e-8, 435.26525504,
        ),
        (
            0.8e-3, 2.0e-3, 18e-6,
            1.4432152717, 2e-8, 0.0178441363, 2e-8, 1291.40977227,
        ),
    ],
)
def test_stripline_finite_thickness_transitions_from_true_dc_to_cohn_skin_limit(
    w, b, t, expected_transition_r, transition_rtol,
    expected_transition_x, transition_x_rtol, expected_skin_weight,
):
    """Known strip thickness anchors DC without changing Cohn's skin limit."""
    sigma = 5.8e7
    freq = Frequency.from_f(jnp.array([0.0, 1e6, 1e13]))
    line = StriplineLine(
        w=w, b=b, t=t, dielectric=2.2,
        conductor=BulkConductor(sigma=sigma), length=1.0,
    )

    resistance = line.immittance(freq).R
    surface_resistance = jnp.real(line.conductor.properties(freq).zs)
    ideal = StriplineLine(
        w=w, b=b, t=t, dielectric=2.2,
        conductor=BulkConductor(sigma=jnp.inf), length=1.0,
    ).immittance(freq)
    internal_reactance = freq.w * (line.immittance(freq).L - ideal.L)
    half_space_impedance = line.conductor.properties(freq).zs * expected_skin_weight

    assert jnp.isclose(resistance[0], 1 / (sigma * w * t), rtol=1e-6)
    # Per-case checkpoints independently evaluated from the documented slab
    # and interpolation equations, not from recorded ParamRF output.
    assert jnp.isclose(resistance[1], expected_transition_r, rtol=transition_rtol)
    assert jnp.isclose(
        internal_reactance[1], expected_transition_x, rtol=transition_x_rtol,
    )
    assert resistance[1] > jnp.real(half_space_impedance[1])
    assert internal_reactance[1] < jnp.imag(half_space_impedance[1])
    assert jnp.isclose(
        resistance[-1], surface_resistance[-1] * expected_skin_weight, rtol=2e-6,
    )


def test_stripline_accepts_a_dispersive_dielectric():
    """A dispersive material needs no stripline-specific code: eps_eff tracks it."""
    from pmrf.models import CohnStriplineFormulation, StriplineLine

    freq = Frequency(start=1.0, stop=20.0, npoints=11, unit="GHz")
    dielectric = DjordjevicSarkar(ep_r=3.9, tand=0.02, f_ref=1e9)
    line = StriplineLine(w=2.655e-3, b=3.2e-3, dielectric=dielectric, length=0.1)

    ep_r = dielectric.properties(freq).ep_r
    quasi_static = CohnStriplineFormulation().quasi_static(
        w=line.w,
        b=line.b,
        t=line.t,
        ep_r=ep_r,
    )

    assert jnp.array_equal(quasi_static.ep_eff, ep_r)
    # The dielectric actually disperses over the band, so this is not vacuous.
    assert jnp.abs(jnp.real(ep_r[0]) - jnp.real(ep_r[-1])) > 1e-3

    s = line.s(freq)
    assert jnp.all(jnp.isfinite(s))


def test_stripline_narrow_strip_uses_the_fringing_correction():
    """Below W/b = 0.35 Cohn's effective width picks up the fringing term."""
    from pmrf.models import StriplineLine

    freq = Frequency(start=10.0, stop=10.0, npoints=1, unit="GHz")
    b = 3.2e-3
    narrow = StriplineLine(
        w=0.2 * b, b=b, dielectric=2.2, conductor=BulkConductor(sigma=jnp.inf), length=0.1
    )

    w_e = 0.2 * b - (0.35 - 0.2) ** 2 * b
    expected = 30 * jnp.pi / jnp.sqrt(2.2) * b / (w_e + 0.441 * b)
    assert jnp.allclose(jnp.real(narrow.zc(freq)), expected, rtol=1e-6)


def test_stripline_high_impedance_branch_is_selected():
    """A narrow strip crosses sqrt(er)*Zc = 120 onto Cohn's other loss fit."""
    from pmrf.models import StriplineLine

    freq = Frequency(start=10.0, stop=10.0, npoints=1, unit="GHz")
    geometry = dict(b=3.2e-3, t=35e-6, dielectric=2.2, length=1.0)
    copper = BulkConductor(sigma=1 / 1.72e-8)

    narrow = StriplineLine(w=0.2e-3, conductor=copper, **geometry)
    wide = StriplineLine(w=2.655e-3, conductor=copper, **geometry)

    assert jnp.sqrt(2.2) * jnp.real(narrow.zc(freq)) > 120
    assert jnp.sqrt(2.2) * jnp.real(wide.zc(freq)) < 120
    # A narrower strip concentrates the current, so it loses more.
    assert jnp.real(narrow.gammaL(freq)) > jnp.real(wide.gammaL(freq)) > 0
