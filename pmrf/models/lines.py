from abc import abstractmethod

from pmrf._numpy import numpy as np
from scipy.constants import c, mu_0, epsilon_0

from pmrf import Model, Parameter
from pmrf._math import evaluate_bernstein_basis, evaluate_power_basis

class RLGCLine(Model):
    """
    Abstract base class for a foundational RLGC line model.

    This class models an RLGC line, while allowing derived classes to specify the forms of the per-unit parameters R, L, G and C.
    A method is provided to calculate the ABCD matrix as a function frequency, which internally calls `self.rlgc(w)`.

    Args:
        length (float): The length of the line.
    """    
    length: Parameter = 1.0

    @abstractmethod
    def rlgc(self, w) -> tuple:
        """The RLGC parameters of the line.
        
        Args:
            w: Angular frequency, specified in radians.

        Returns:
            tuple: The R, L, G and C line parameters, in that order.
        """
        raise NotImplementedError("'rlgc' must be implemented in the derived class")

    def a(self, w):
        R, L, G, C = self.rlgc(w)
        gamma = np.sqrt((R + 1j*w*L) * (G + 1j*w*C))
        Zc = np.sqrt((R + 1j*w*L) / (G + 1j*w*C))

        gL = gamma*self.length
        
        a = np.array([
            [np.cosh(gL), Zc * np.sinh(gL)],
            [1 / Zc * np.sinh(gL), np.cosh(gL)]
        ]).transpose(2, 0, 1)

        return a
    
class ConstantRLGCLine(RLGCLine):
    """An RLGC line with constant per-unit parameters as a function of frequency.

    Args:
        R (float): Per-unit resistance.
        L (float): Per-unit inductance.
        C (float): Per-unit capacitance.
        G (float): Per-unit conductance.
        length (float, optional): The length of the line. Default to 1.0.
    """
    R: Parameter = 0.0
    L: Parameter = 280e-9
    G: Parameter = 0.0
    C: Parameter = 90e-12,

    def rlgc(self, w) -> tuple:
        return self.R, self.L, self.G, self.C
    
class DatasheetCoaxial(RLGCLine):
    """
    A coaxial line defined by constants typically found on datasheets.

    Additionally, the dielectric constant can be sloped if `epr_slope` is passed, with `freq_bounds` allowing
    to specify frequency bounds different to the model's bounds for this slope.

    Args:
        zn (float, optional): Nominal characteristic impedance. Defaults to 50.0.
        epr (float, optional): Dielectric constant (1 / vf**2). Defaults to 1.0.
        epr_slope (float, optional): Slope of the dielectric constant. Defaults to 0.0.
        k1 (float, optional): Skin effect loss (~ sqrt(w)). Defaults to 0.0.
        k2 (float, optional): Dielectric loss (~ w). Defaults to 0.0.
        length (float, optional): The length of the line. Default to 1.0.
        loss_coeffs_normalized (bool, optional): Generally, loss coefficients `k1` and `k2` are in terms of datasheet units (100m and MHz). If True, units should instead be dB/1m/sqrt(rad) and dB/1m/rad. Defaults to False.
        freq_bounds (tuple, optional): The min and max normalizing bounds for the frequency slope. Defaults to None, in which case the minimum and maximum bounds of the frequency are used.
    """
    zn: Parameter = 50.0
    epr: Parameter = 1.0
    epr_slope: Parameter = 0.0
    k1: Parameter = 0.0
    k2: Parameter = 0.0

    loss_coeffs_normalized: bool = False
    freq_bounds: tuple = None

    def rlgc(self, w):
        zn, k1, k2 = self.zn, self.k1, self.k2

        if self.epr_slope == 0:
            epr = np.ones(w.shape[0]) * self.epr
        else:
            if not self.freq_bounds is None:
                w_start, w_stop = self.freq_bounds
            else:
                w_start, w_stop = w[0], w[-1]

            wn = (w - w_start) / (w_stop - w_start)            
            epr += self.epr_slope * wn
        
        if not self.loss_coeffs_normalized:
            k1_norm = k1 * (1.0 / (100 * np.sqrt(2*np.pi * 10**6)))
            k2_norm = k2 * (1.0 / (100 * 2*np.pi * 10**6))
        else:
            k1_norm = k1
            k2_norm = k2

        sqrt_w = np.sqrt(w)
        dBtoNeper = np.log(10) / 20
        alpha_c = k1_norm * dBtoNeper * sqrt_w
        alpha_d = k2_norm * dBtoNeper * w
        sqrt_epr = np.sqrt(epr)
        
        R = 2*zn * alpha_c
        L = (zn * sqrt_epr) / c
        G = 2/zn * alpha_d
        C = (sqrt_epr) / (zn * c)
        return R, L, G, C
        

class PhysicalCoaxial(RLGCLine):
    """
    A coaxial line defined directly by its physical properties (geometric and material).

    A number of the parameters allow a `xxx_model` flag alongside, which specifies the form of the parameter as a function of frequency.
    In this case, for some models, the parameter may be a list of coefficients as opposed to a single value. Currently, the following models are provided:

        'constant' (default): The parameter is represented by a single value across frequency.
        'ppoly': The parameter and its coefficients specify a polynomial in the power basis across frequency.
        'bpoly': The parameter and its coefficients specify a polynomial in the Bernstein basis across frequency.

    Args:
        din (float, optional): Inner diameter. Defaults to 1.12e-3.
        dout (float, optional): Outer diameter. Defaults to 3.2e-3.
        epr (float | list[float], optional): Relative dielectric permittivity. Can be a list of coefficients, whose meaning is specified by `epr_model`. Defaults to 1.0.
        mur (float | list[float], optional): Relative dielectric permeability. Can be a list of coefficients, whose meaning is specified by `mur_model`. Defaults to 1.0.
        tand (float | list[float], optional): Loss tangent. Can be a list of coefficients, whose meaning is specified by `tand_model`. Defaults to 0.0.
        rho (float | list[float], optional): Conductor resistivity. Can be a list of coefficients, whose meaning is specified by `rho_model`. Is ignored if `separate_rho == True` is passed, in which case `rhoin` and `rhoout` are used. Defaults to 1.68e-8.
        rhoin (float | list[float], optional): Inner conductor resistivity. Can be a list of coefficients, whose meaning is specified by `rho_model`. Only used if `separate_rho == True` is passed. Defaults to 1.68e-8.
        rhoout (float | list[float], optional): Outer conductor resistivity. Can be a list of coefficients, whose meaning is specified by `rho_model`. Only used if `separate_rho == True` is passed. Defaults to 1.68e-8.
        length (float, optional): The length of the line. Default to 1.0.
        epr_model (str, optional): The model for the dielectric permittivity. See the documentation on models above. Defaults to 'constant'.
        mur_model (str, optional): The model for the dielectric permeability. See the documentation on models above. Defaults to 'constant'.
        tand_model (str, optional): The model for the loss tangent. See the documentation on models above. Defaults to 'constant'.
        rho_model (str, optional): The model for the conductor resistivity. See the documentation on models above. Defaults to 'constant'.
        freq_bounds (tuple, optional): The min and max normalizing bounds for the frequency slope. Defaults to None, in which case the minimum and maximum bounds of the frequency are used.
        separate_rho (bool, optional): Whether or not the conductor resistivity should be modelled as seprate values for the inner and outer conductors. Defaults to False.
        neglect_skin_inductance (bool, optional): Specifies whether to neglect the incremental skin inductance term internally. Defaults to False.
    """
    # Main parameters
    din: Parameter = 1.12e-3
    dout: Parameter = 3.2e-3
    epr: Parameter = 1.0
    mur: Parameter = 1.0
    tand: Parameter = 0.0
    rho: Parameter = 1.68e-8
    
    # Optional parameters
    rhoin: Parameter = 1.68e-8
    rhoout: Parameter = 1.68e-8

    # Hyperparameters
    epr_model: str = 'constant'
    mur_model: str = 'constant'
    tand_model: str = 'constant'
    rho_model: str = 'constant'
    freq_bounds: tuple = None
    separate_rho: bool = False
    neglect_skin_inductance: bool = False
        
    def interpolated(self, param: str, w):
        if param.startswith('rho'):
            model = str(getattr(self, f'rho_model'))
        else:
            model = str(getattr(self, f'{param}_model'))
        
        if model == 'constant':
            value = getattr(self, param) * np.ones(w.shape[0])
        else:
            coeffs = getattr(self, param)
            lb, ub = self.freq_bounds or (w[0], w[-1])
            if model.startswith('ppoly'):
                value = evaluate_power_basis(w, coeffs, lb, ub)
            else:
                value = evaluate_bernstein_basis(w, coeffs, lb, ub)
                
        return value
            
    def epr_w(self, w, x='hello'):
        return self.interpolated('epr', w)
    
    def tand_w(self, w):
        return self.interpolated('tand', w)
    
    def mur_w(self, w):
        return self.interpolated('mur', w)
    
    def rho_w(self, w):
        return self.interpolated('rho', w)
    
    def rhoin_w(self, w):
        return self.interpolated('rhoin', w) if self.separate_rho else self.rho_w(w)
    
    def rhoout_w(self, w):
        return self.interpolated('rhout', w) if self.separate_rho else self.rho_w(w)
    
    def eps_w(self, w):
        return epsilon_0 * self.epr_w(w) * (1 - 1j * self.tand_w(w))
    
    def mu_w(self, w):
        return mu_0 * self.mur_w(w)
    
    def L_prime(self, w):
        a, b = self.din / 2, self.dout / 2
        lnbOvera = np.log(b/a)
        return self.mu_w(w) / (2 * np.pi) * lnbOvera
    
    def C_prime(self, w):
        a, b = self.din / 2, self.dout / 2
        lnbOvera = np.log(b/a)
        return 2 * np.pi * np.real(self.eps_w(w)) / lnbOvera
    
    def G_diel(self, w):
        a, b = self.din / 2, self.dout / 2
        lnbOvera = np.log(b/a)
        return 2 * np.pi * w * -np.imag(self.eps_w(w)) / lnbOvera
        
    def R_skin(self, w):
        return np.real(self.Z_skin(w))
    
    def L_skin(self, w):
        return np.imag(self.Z_skin(w)) / w
    
    def Z_skin(self, w):
        a = self.din / 2
        b = self.dout / 2
        mu = self.mu_w(w)
        sigma_a = 1 / self.rhoin_w(w)
        sigma_b = 1 / self.rhoout_w(w)
        
        L_skin_a = (1 / (2 * np.pi * a)) * np.sqrt(mu / (2 * w * sigma_a))
        L_skin_b = (1 / (2 * np.pi * b)) * np.sqrt(mu / (2 * w * sigma_b))
        L_skin = L_skin_a + L_skin_b
                
        R_skin_a = (1 / (2 * np.pi * a)) * np.sqrt(w * mu / (2 * sigma_a))
        R_skin_b = (1 / (2 * np.pi * b)) * np.sqrt(w * mu / (2 * sigma_b))
        R_skin = R_skin_a + R_skin_b            
        
        return R_skin + 1j * w * L_skin
    
    def rlgc(self, w):
        # print(f'PhysicalCoax.compute: {self.name}')
        # Formulae from 'Frederick M. Tesche - A Simple Model for the Line Parameters of a Lossy Coaxial Cable Filled With a Nondispersive Dielectric'
        # as well as Pozar for G term
        if not self.neglect_skin_inductance:
            L = self.L_prime(w) + self.L_skin(w)
        else:
            L = self.L_prime(w)
        
        C = self.C_prime(w)
        G = self.G_diel(w)
        R = self.R_skin(w)
        
        return R, L, G, C