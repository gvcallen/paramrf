"""
Curated wideband validation of ParamRF lines against the scikit-rf media.

This suite is the external reference point for the materials and line rework.
It sweeps the kilohertz region through 40 GHz, so every case exercises the
low-frequency conductor transition — where the series impedance is resistive
and the skin effect has not yet taken over — as well as microwave behaviour,
and compares both the internal electrical quantities and the public
S-parameters.

**Scope.** ParamRF carries permittivity complex throughout, the ADS/AWR
convention. Only the scikit-rf modes matching that convention are used:
``compatibility_mode=None`` for :class:`~skrf.media.MLine`, and the complex
``dielectric={'ep_r': ...}`` filling for :class:`~skrf.media.Coaxial`. QUCS
compatibility is deliberately out of scope — ParamRF does not implement the
real-permittivity QUCS path, so there is nothing to validate against it.

Stripline is not covered here: scikit-rf has no stripline medium. It stays
validated against Cohn's published values and against analytic identities in
``test_lines.py``.

**Coverage.** Coaxial: inner/outer diameter combinations from 0.51/1.68 mm to
0.3/7.0 mm, lossless through lossy fillings, a filling with static bulk
conductivity, perfect through stainless-steel conductors, and both constant and
Djordjevic-Sarkar permittivity. Microstrip: w/h from 0.1 to 10, relative
permittivity from 2.2 to 10, loss tangents from 0 to 0.02, zero and finite
conductor thickness, modal dispersion on and off, both Wheeler and
Hammerstad-Jensen quasi-static formulations, and both constant and
Djordjevic-Sarkar permittivity.

**Curation.** The cases are chosen to cover every modelling axis plus selected
cross-axis stress points, not the Cartesian product of them. Each case records
what it is for in its ``purpose``, and carries its own tolerance per quantity.
Any tolerance above floating-point noise carries a note naming which side is
approximating and why.

Runtime budget: 121 log-spaced frequency points per case, thirteen cases, two
tests each. The whole file runs in under half a minute on an unloaded machine,
most of that JAX compilation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.constants import epsilon_0

from pmrf.frequency import Frequency
from pmrf.materials import BulkConductor, ConstantDielectric, DjordjevicSarkar
from pmrf.models import (
    CoaxialLine,
    HammerstadJensenMicrostripFormulation,
    KirschningJansenMicrostripDispersion,
    MicrostripLine,
    WheelerMicrostripFormulation,
)

# scikit-rf is an optional dependency, so the whole module skips without it.
skrf = pytest.importorskip("skrf")
from skrf.media import Coaxial, MLine  # noqa: E402

#: Wideband axis shared by every case: 1 kHz to 40 GHz, log spaced so the
#: decades below the skin-effect onset get as many points as the microwave end.
WIDEBAND = Frequency.from_f(jnp.geomspace(1e3, 40e9, 121))

#: Copper and stainless steel: the ordinary and the deliberately lossy metal.
RHO_COPPER = 1.68e-8
RHO_STEEL = 7.2e-7

#: Djordjevic-Sarkar relaxation band. These are material constants; scikit-rf
#: spells the reference frequency ``f_epr_tand``.
DS_BAND = dict(f_low=1e3, f_high=1e12, f_ref=1e9)


@dataclass(frozen=True)
class Case:
    """One curated comparison: a ParamRF line, its scikit-rf twin, tolerances."""

    #: pytest id, also the human-readable name of the case
    id: str
    #: why this combination is in the matrix
    purpose: str
    #: builds the ParamRF line of a given physical length
    line: Callable[[float], object]
    #: builds the scikit-rf medium on a scikit-rf frequency axis, optionally
    #: with a port impedance to renormalize into
    media: Callable[..., object]
    #: physical length in metres used for the S-parameter comparison
    length: float
    #: port reference impedance used for the S-parameter comparison
    z0: float
    #: per-quantity ``(rtol, atol)``; the keys name the quantities compared
    tol: dict[str, tuple[float, float]]
    #: what explains any tolerance above the floating-point noise floor
    notes: dict[str, str] = field(default_factory=dict)
    #: lowest frequency at which a quantity is comparable at all, where the two
    #: implementations model different things below it. Needs a note.
    f_min: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        # `notes` and `f_min` are keyed by the same quantity names as `tol`, and
        # a typo in either would silently drop the explanation or the floor it
        # was meant to carry rather than fail.
        for name, keyed in (("notes", self.notes), ("f_min", self.f_min)):
            unknown = set(keyed) - set(self.tol)
            assert not unknown, f"{self.id}: {name} keys not in tol: {unknown}"


def _assert_close(actual, desired, quantity, case):
    """Compare one quantity over the sweep against its own recorded tolerance."""
    rtol, atol = case.tol[quantity]
    f = np.asarray(WIDEBAND.f)
    band = f >= case.f_min.get(quantity, 0.0)
    actual = np.asarray(actual)[band]
    desired = np.broadcast_to(np.asarray(desired), f.shape + np.shape(desired)[1:])[band]

    error = np.abs(actual - desired)
    scale = rtol * np.abs(desired) + atol
    excess = error - scale
    worst = np.unravel_index(int(np.argmax(excess)), excess.shape)
    assert np.all(error <= scale), (
        f"{case.id}: {quantity} disagrees with scikit-rf. "
        f"Worst point f = {f[band][worst[0]]:.4g} Hz: "
        f"ParamRF {actual[worst]!r} vs scikit-rf {desired[worst]!r} "
        f"(rtol={rtol}, atol={atol}). {case.notes.get(quantity, '')}"
    )


# -----------------------------------------------------------------------------
# Coaxial
# -----------------------------------------------------------------------------

def _coax_ep_r(ep_r, tand=0.0, sigma=0.0, f=None):
    r"""The complex filling permittivity in the convention both sides share.

    ParamRF's :class:`~pmrf.materials.ConstantDielectric` returns
    $\varepsilon_r(1 - j\tan\delta) - j\sigma/(\omega\varepsilon_0)$. scikit-rf's
    ``epsilon_r``/``tan_delta`` pair covers only the first term, so the static
    conductivity is handed over through the complex ``dielectric`` dict, which is
    scikit-rf's own documented path for a filling it cannot otherwise describe.
    """
    eps = ep_r * (1 - 1j * tand) * np.ones_like(f)
    if sigma:
        eps = eps - 1j * sigma / (2 * np.pi * f * epsilon_0)
    return eps


def _skrf_djordjevic_sarkar(*, ep_r, tand):
    """scikit-rf's own Djordjevic-Svensson permittivity, as a function of f.

    Read off a throwaway :class:`~skrf.media.MLine`, whose geometry is
    irrelevant to the filling: the dielectric stage runs before any of it. This
    is the only place scikit-rf implements the model, and borrowing it here is
    what makes the coaxial case a cross-check of ParamRF's
    :class:`~pmrf.materials.DjordjevicSarkar` rather than of itself.
    """
    def ep_r_of_f(f):
        return MLine(
            skrf.Frequency.from_f(f, unit="Hz"), ep_r=ep_r, tand=tand,
            diel="djordjevicsvensson", f_low=DS_BAND["f_low"],
            f_high=DS_BAND["f_high"], f_epr_tand=DS_BAND["f_ref"],
        ).ep_r_f

    return ep_r_of_f


def _coax_pair(d_in, d_out, dielectric, rho, ep_r_of_f):
    """A ParamRF/scikit-rf builder pair for one coaxial geometry."""
    def line(length):
        return CoaxialLine(
            d_in=d_in, d_out=d_out, dielectric=dielectric,
            conductor=BulkConductor(rho=rho), length=length,
        )

    def media(freq, z0_port=None):
        return Coaxial(
            freq, Dint=d_in, Dout=d_out, model="tesche", z0_port=z0_port,
            sigma=np.inf if rho == 0.0 else 1 / rho,
            dielectric={"ep_r": ep_r_of_f(np.asarray(freq.f))},
        )

    return {"line": line, "media": media}


#: Tesche's equivalent circuit is implemented independently on both sides from
#: the same paper, so the per-unit-length quantities agree to round-off. Zc and
#: gamma go through a square root of their ratio and product, which costs about
#: a decade of relative precision at the low-frequency end where R >> wL.
_COAX_EXACT = {
    "R": (1e-9, 1e-14), "L": (1e-9, 0.0), "G": (1e-9, 0.0), "C": (1e-9, 0.0),
    "zc": (1e-7, 0.0), "alpha": (1e-7, 0.0), "beta": (1e-7, 0.0),
    "s": (0.0, 1e-9),
}

COAX_CASES = [
    Case(
        id="air_line_perfect_conductor",
        purpose="Boundary case: lossless everywhere, so only the geometry and "
                "the external inductance are exercised.",
        **_coax_pair(3.04e-3, 7.00e-3, ConstantDielectric(ep_r=1.0), 0.0,
                    lambda f: _coax_ep_r(1.0, f=f)),
        length=0.1, z0=50.0, tol=_COAX_EXACT,
    ),
    Case(
        id="rg58_ptfe_copper",
        purpose="Nominal 50 ohm cable: moderate diameter ratio, low-loss "
                "dielectric, ordinary copper conductors.",
        **_coax_pair(0.9e-3, 2.95e-3, ConstantDielectric(ep_r=2.25, tand=1e-3),
                    RHO_COPPER, lambda f: _coax_ep_r(2.25, 1e-3, f=f)),
        length=0.5, z0=50.0, tol=_COAX_EXACT,
    ),
    Case(
        id="semirigid_small_diameters_steel",
        purpose="Stress: small diameters and a stainless conductor push the "
                "conductor loss and the dc-to-skin transition high.",
        **_coax_pair(0.51e-3, 1.68e-3, ConstantDielectric(ep_r=2.1, tand=2e-4),
                    RHO_STEEL, lambda f: _coax_ep_r(2.1, 2e-4, f=f)),
        length=0.25, z0=50.0, tol=_COAX_EXACT,
    ),
    Case(
        id="foam_large_diameter_ratio",
        purpose="Boundary: a large b/a with a near-air foam filling gives a "
                "high characteristic impedance.",
        **_coax_pair(0.3e-3, 7.0e-3, ConstantDielectric(ep_r=1.2, tand=1e-4),
                    RHO_COPPER, lambda f: _coax_ep_r(1.2, 1e-4, f=f)),
        length=0.2, z0=75.0, tol=_COAX_EXACT,
    ),
    Case(
        id="conductive_filling",
        purpose="A filling with static bulk conductivity, which keeps a finite "
                "shunt conductance down to dc instead of vanishing with omega.",
        **_coax_pair(0.9e-3, 2.95e-3,
                    ConstantDielectric(ep_r=2.25, tand=1e-3, sigma=1e-3),
                    RHO_COPPER, lambda f: _coax_ep_r(2.25, 1e-3, 1e-3, f=f)),
        length=0.1, z0=50.0, tol=_COAX_EXACT,
    ),
    Case(
        id="djordjevic_sarkar_filling",
        purpose="Causal wideband dielectric: the permittivity itself is now "
                "frequency dependent, cross-checked against scikit-rf's own "
                "Djordjevic-Svensson implementation.",
        **_coax_pair(
            0.9e-3, 2.95e-3,
            DjordjevicSarkar(ep_r=4.3, tand=0.02, f_low=DS_BAND["f_low"],
                             f_high=DS_BAND["f_high"], f_ref=DS_BAND["f_ref"]),
            RHO_COPPER,
            _skrf_djordjevic_sarkar(ep_r=4.3, tand=0.02),
        ),
        length=0.1, z0=50.0, tol=_COAX_EXACT,
    ),
]


@pytest.mark.parametrize("case", COAX_CASES, ids=lambda case: case.id)
def test_coaxial_matches_skrf_electrical_quantities(case):
    """Per-unit-length R, L, G, C plus Zc and gamma, over the full sweep."""
    line = case.line(case.length)
    media = case.media(WIDEBAND.to_skrf())
    imm = line.immittance(WIDEBAND)

    _assert_close(imm.R, media.R, "R", case)
    _assert_close(imm.L, media.L, "L", case)
    _assert_close(imm.G, media.G, "G", case)
    _assert_close(imm.C, media.C, "C", case)

    zc, gamma_length = line.zc_and_gammaL(WIDEBAND)
    gamma = gamma_length / line.length
    _assert_close(zc, media.z0_characteristic, "zc", case)
    # Attenuation and phase are compared separately: beta dwarfs alpha across
    # most of the sweep, so a single complex comparison would hide the loss.
    _assert_close(np.real(gamma), np.real(media.gamma), "alpha", case)
    _assert_close(np.imag(gamma), np.imag(media.gamma), "beta", case)


@pytest.mark.parametrize("case", COAX_CASES, ids=lambda case: case.id)
def test_coaxial_matches_skrf_s_parameters(case):
    """Formulation, electrical length and renormalization, end to end."""
    line = case.line(case.length)
    media = case.media(WIDEBAND.to_skrf(), z0_port=case.z0)

    expected = media.line(case.length, "m").s
    _assert_close(line.s(WIDEBAND, z0=case.z0), expected, "s", case)


# -----------------------------------------------------------------------------
# Microstrip
# -----------------------------------------------------------------------------

def _microstrip_pair(*, w, h, t, ep_r, tand, rho, dispersion, diel, formulation,
                     model):
    """A ParamRF/scikit-rf builder pair for one microstrip cross-section."""
    def line(length):
        dielectric = (
            ConstantDielectric(ep_r=ep_r, tand=tand) if diel == "frequencyinvariant"
            else DjordjevicSarkar(ep_r=ep_r, tand=tand, f_low=DS_BAND["f_low"],
                                  f_high=DS_BAND["f_high"], f_ref=DS_BAND["f_ref"])
        )
        return MicrostripLine(
            w=w, h=h, t=t, dielectric=dielectric,
            conductor=BulkConductor(rho=rho),
            formulation=formulation(),
            dispersion=None if dispersion is None else dispersion(),
            length=length,
        )

    def media(freq, z0_port=None):
        return MLine(
            freq, w=w, h=h, t=t, ep_r=ep_r, tand=tand, rho=rho, rough=0.0,
            z0_port=z0_port,
            model=model, disp="none" if dispersion is None else "kirschningjansen",
            diel=diel, f_low=DS_BAND["f_low"], f_high=DS_BAND["f_high"],
            f_epr_tand=DS_BAND["f_ref"], compatibility_mode=None,
        )

    return {"line": line, "media": media}


def _ep_eff(line, freq):
    """The complex effective permittivity a ParamRF microstrip line ends up with."""
    ep_r = line.substrate.dielectric.properties(freq).ep_r
    quasi_static = line.formulation.quasi_static(
        w=line.w, h=line.substrate.h, t=line.substrate.t, ep_r=ep_r
    )
    if line.dispersion is None:
        return quasi_static.ep_eff
    ep_eff, _ = line.dispersion.disperse(
        freq, ep_eff_0=quasi_static.ep_eff, zc_0=quasi_static.zc, ep_r=ep_r,
        w_eff=quasi_static.w_eff, h=line.substrate.h,
    )
    return ep_eff


# Every microstrip case sits on the same set of documented differences between
# the two implementations, so the notes are written once and shared.
_MICROSTRIP_NOTES = {
    "alpha": (
        "Dielectric loss is modelled differently on the two sides, and this is "
        "the expected consequence rather than an error. ParamRF evaluates the "
        "quasi-static formulation at a complex permittivity, so its "
        "attenuation is (w/2c)*ep_r*tand*(d ep_eff/d ep_r)/sqrt(ep_eff), the "
        "exact derivative of the model actually in use. scikit-rf instead adds "
        "the classical filling-factor term, pi*ep_r*(ep_eff-1)*tand/"
        "((ep_r-1)*sqrt(ep_eff)*lambda_0), which assumes ep_eff is linear in "
        "ep_r. Hammerstad-Jensen is not linear in ep_r, so the two differ by "
        "up to about ten percent. The conductor term is identical on both "
        "sides: Re(Zs)/(Re(Zc)*w) times Wheeler's current-distribution factor."
    ),
    "beta": (
        "ParamRF takes beta = Im(w*sqrt(ep_eff)/c) from the complex effective "
        "permittivity; scikit-rf builds it from Re(ep_eff) alone. The two "
        "differ at second order in the loss tangent."
    ),
    "s": (
        "The accumulated consequence of the attenuation difference described "
        "for alpha, over the physical length of this case."
    ),
}

#: Hammerstad-Jensen and Kirschning-Jansen are implemented independently on
#: both sides from the same papers, and agree on the lossless quantities to
#: round-off. Only the loss splits above separate them.
_MICROSTRIP_HJ = {
    "ep_eff": (1e-12, 0.0), "zc": (1e-12, 0.0),
    "alpha": (1e-1, 1e-12), "beta": (1e-4, 0.0), "s": (0.0, 5e-3),
}

MICROSTRIP_CASES = [
    Case(
        id="narrow_low_er_lossless_quasi_static",
        purpose="Boundary: u = w/h = 0.1, the narrow end of Hammerstad-Jensen, "
                "lossless and with modal dispersion disabled, so the "
                "quasi-static solution stands alone.",
        **_microstrip_pair(w=0.1e-3, h=1e-3, t=None, ep_r=2.2, tand=0.0, rho=0.0,
                           dispersion=None, diel="frequencyinvariant",
                           formulation=HammerstadJensenMicrostripFormulation,
                           model="hammerstadjensen"),
        length=5e-3, z0=50.0,
        tol={"ep_eff": (1e-12, 0.0), "zc": (1e-12, 0.0),
             "alpha": (0.0, 1e-12), "beta": (1e-12, 0.0), "s": (0.0, 1e-8)},
        notes={
            "s": "Zc and gamma agree to round-off here, so the residual is "
                 "scikit-rf's own renormalization noise: its S matrix for this "
                 "202 ohm line seen at 50 ohm ports comes back very slightly "
                 "non-reciprocal, S21 != S12 by a few times 1e-9, where "
                 "ParamRF's is symmetric by construction.",
        },
    ),
    Case(
        id="wide_low_er_lossy_quasi_static",
        purpose="Boundary: u = 10, the wide end, with a lossy dielectric, a "
                "finite conductor thickness and modal dispersion disabled.",
        **_microstrip_pair(w=10e-3, h=1e-3, t=35e-6, ep_r=2.2, tand=0.002,
                           rho=RHO_COPPER, dispersion=None,
                           diel="frequencyinvariant",
                           formulation=HammerstadJensenMicrostripFormulation,
                           model="hammerstadjensen"),
        length=5e-3, z0=50.0,
        tol={"ep_eff": (1e-12, 0.0), "zc": (5e-3, 0.0),
             "alpha": (1e-1, 1e-12), "beta": (5e-3, 0.0), "s": (0.0, 5e-3)},
        f_min={"zc": 1e9, "alpha": 1e9, "beta": 1e9, "s": 1e9},
        notes={
            "zc": "Finding, not a tolerance to widen. With modal dispersion "
                  "disabled ParamRF takes the quasi-static immittance path and "
                  "reports the true RLGC characteristic impedance sqrt(Z/Y), "
                  "which rises as f**-0.25 once the sheet impedance of the "
                  "conductor exceeds w*L. "
                  "scikit-rf's MLine is a (Zc, gamma) medium whose "
                  "z0_characteristic is the quasi-static value at every "
                  "frequency and never sees the conductor resistance. The two "
                  "definitions only converge above the transition, so the "
                  "comparison starts at 1 GHz; the low-frequency limit is "
                  "checked against theory in "
                  "test_quasi_static_microstrip_zc_follows_the_rlgc_limit.",
            "alpha": _MICROSTRIP_NOTES["alpha"] + " On this quasi-static path "
                     "the conductor term differs too: ParamRF charges the sheet "
                     "impedance over the effective width, 2*Zs/w_eff, where "
                     "scikit-rf applies Wheeler's incremental-inductance rule.",
            "beta": _MICROSTRIP_NOTES["beta"] + " On this path beta also "
                    "carries the conductor resistance through sqrt(Z*Y), which "
                    "is what the 1 GHz floor excludes.",
            "s": _MICROSTRIP_NOTES["s"],
        },
    ),
    Case(
        id="nominal_fr4_dispersive",
        purpose="Nominal board: u = 1 on FR-4, finite copper thickness and "
                "Kirschning-Jansen dispersion, the everyday configuration.",
        **_microstrip_pair(w=1e-3, h=1e-3, t=35e-6, ep_r=4.3, tand=0.02,
                           rho=RHO_COPPER,
                           dispersion=KirschningJansenMicrostripDispersion,
                           diel="frequencyinvariant",
                           formulation=HammerstadJensenMicrostripFormulation,
                           model="hammerstadjensen"),
        length=5e-3, z0=50.0, tol=_MICROSTRIP_HJ, notes=_MICROSTRIP_NOTES,
    ),
    Case(
        id="narrow_high_er_lossy_stress",
        purpose="Cross-axis stress: the narrow end of the width range on the "
                "high end of the permittivity range, with the largest loss "
                "tangent and a deliberately resistive conductor, all at once.",
        **_microstrip_pair(w=0.1e-3, h=1e-3, t=18e-6, ep_r=10.0, tand=0.02,
                           rho=1e-6,
                           dispersion=KirschningJansenMicrostripDispersion,
                           diel="frequencyinvariant",
                           formulation=HammerstadJensenMicrostripFormulation,
                           model="hammerstadjensen"),
        length=2e-3, z0=50.0, notes=_MICROSTRIP_NOTES,
        # The most attenuating case in the matrix, so the documented
        # dielectric-loss difference shows up largest in S21.
        tol={**_MICROSTRIP_HJ, "s": (0.0, 1e-2)},
    ),
    Case(
        id="wide_high_er_zero_thickness_dispersive",
        purpose="u = 10 on a high-permittivity substrate with unspecified "
                "conductor thickness and a lossless conductor, so the "
                "dielectric loss stands alone. (A nonzero rho here would pit "
                "ParamRF's unconditional Wheeler correction against "
                "scikit-rf's policy of zeroing conductor loss whenever t is "
                "unspecified -- a difference in missing-input policy, not "
                "physics, already covered by "
                "test_microstrip_line_default_construction_has_conductor_loss "
                "in test_lines.py.)",
        **_microstrip_pair(w=10e-3, h=1e-3, t=None, ep_r=10.0, tand=0.002,
                           rho=0.0,
                           dispersion=KirschningJansenMicrostripDispersion,
                           diel="frequencyinvariant",
                           formulation=HammerstadJensenMicrostripFormulation,
                           model="hammerstadjensen"),
        length=2e-3, z0=75.0, tol=_MICROSTRIP_HJ, notes=_MICROSTRIP_NOTES,
    ),
    Case(
        id="djordjevic_sarkar_fr4_dispersive",
        purpose="Causal wideband substrate: material dispersion and modal "
                "dispersion active together, cross-checked against scikit-rf's "
                "own Djordjevic-Svensson implementation.",
        **_microstrip_pair(w=1e-3, h=1e-3, t=35e-6, ep_r=4.3, tand=0.02,
                           rho=RHO_COPPER,
                           dispersion=KirschningJansenMicrostripDispersion,
                           diel="djordjevicsvensson",
                           formulation=HammerstadJensenMicrostripFormulation,
                           model="hammerstadjensen"),
        length=5e-3, z0=50.0, tol=_MICROSTRIP_HJ, notes=_MICROSTRIP_NOTES,
    ),
    Case(
        id="wheeler_formulation_nominal",
        purpose="Formulation axis: ParamRF's Wheeler quasi-static solution "
                "against scikit-rf's, on a nominal lossless cross-section.",
        **_microstrip_pair(w=3e-3, h=1.6e-3, t=None, ep_r=4.3, tand=0.0, rho=0.0,
                           dispersion=None, diel="frequencyinvariant",
                           formulation=WheelerMicrostripFormulation,
                           model="wheeler"),
        # A short line: a three percent difference in the effective
        # permittivity is a phase error that grows with electrical length, and
        # at 40 GHz a 50 mm line would wrap it past a radian.
        length=1e-3, z0=50.0,
        tol={"ep_eff": (3e-2, 0.0), "zc": (5e-3, 0.0),
             "alpha": (0.0, 1e-12), "beta": (2e-2, 0.0), "s": (0.0, 3e-2)},
        notes={
            "ep_eff": "ParamRF implements the Hammerstad simplification of "
                      "Wheeler's synthesis formula; scikit-rf implements "
                      "Wheeler's own closed form. They are two different "
                      "approximations to the same quasi-static problem and "
                      "agree to about three percent in effective "
                      "permittivity.",
            "zc": "The same two closed forms, agreeing to half a percent in "
                  "characteristic impedance.",
            "beta": "Follows the effective-permittivity difference above.",
            "s": "Follows the impedance and phase differences above.",
        },
    ),
]


# scikit-rf warns that its conductor-loss formula is invalid where the metal is
# thinner than three skin depths. That is true of every finite thickness at the
# kilohertz end of this sweep, and is exactly the regime the suite is here to
# cover, so the warning is expected rather than suppressed silently.
_thin_metal_warning = pytest.mark.filterwarnings(
    "ignore:Conductor loss calculation invalid:RuntimeWarning"
)


@_thin_metal_warning
@pytest.mark.parametrize("case", MICROSTRIP_CASES, ids=lambda case: case.id)
def test_microstrip_matches_skrf_electrical_quantities(case):
    """Effective permittivity, characteristic impedance, attenuation and phase.

    scikit-rf's :class:`~skrf.media.MLine` is a ``DefinedGammaZ0`` medium and
    exposes no per-unit-length R, L, G, C, so those are compared only for the
    coaxial line, where both sides define them.
    """
    line = case.line(case.length)
    media = case.media(WIDEBAND.to_skrf())

    zc, gamma_length = line.zc_and_gammaL(WIDEBAND)
    gamma = gamma_length / line.length

    # The effective permittivity is what the phase constant is built from,
    # compared in its own right so a phase failure stays distinguishable from a
    # loss failure.
    _assert_close(_ep_eff(line, WIDEBAND), media.ep_reff_f, "ep_eff", case)
    _assert_close(zc, media.z0_characteristic, "zc", case)
    _assert_close(np.real(gamma), np.real(media.gamma), "alpha", case)
    _assert_close(np.imag(gamma), np.imag(media.gamma), "beta", case)


@_thin_metal_warning
@pytest.mark.parametrize("case", MICROSTRIP_CASES, ids=lambda case: case.id)
def test_microstrip_matches_skrf_s_parameters(case):
    """Formulation, dispersion, electrical length and renormalization together."""
    line = case.line(case.length)
    media = case.media(WIDEBAND.to_skrf(), z0_port=case.z0)

    expected = media.line(case.length, "m").s
    _assert_close(line.s(WIDEBAND, z0=case.z0), expected, "s", case)


def test_quasi_static_microstrip_zc_follows_the_rlgc_limit():
    r"""The low-frequency Zc that scikit-rf's MLine cannot express.

    Once the sheet impedance of the conductor outgrows $j\omega L$, the series
    impedance of this line is $2Z_s/W_{eff}$, which is
    $(1 + j)R_s \propto \sqrt{f}$. So
    $$Z_c = \sqrt{Z/Y} \to \sqrt{\frac{(1 + j)R_s}{j\omega C}},$$
    whose magnitude rises as $f^{-1/4}$ and whose phase settles at
    $(45 - 90)/2 = -22.5$ degrees. scikit-rf's MLine reports the quasi-static
    impedance there instead, because its ``z0_characteristic`` never sees the
    conductor at all. That is the difference the ``f_min`` floor on
    ``wide_low_er_lossy_quasi_static`` excludes, checked here against theory
    rather than against the other implementation.
    """
    case = next(c for c in MICROSTRIP_CASES if c.id == "wide_low_er_lossy_quasi_static")
    line = case.line(case.length)

    low = Frequency.from_f(jnp.geomspace(1e-3, 1e1, 9))
    zc = line.zc_and_gammaL(low)[0]

    scaled = jnp.abs(zc) * low.f ** 0.25
    assert jnp.allclose(scaled, scaled[0], rtol=1e-2)
    assert jnp.allclose(jnp.angle(zc, deg=True), -22.5, atol=1.0)
