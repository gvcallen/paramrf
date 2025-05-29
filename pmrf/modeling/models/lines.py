import numpy as np
import skrf as rf
from scipy.interpolate import CubicSpline
from scipy.constants import c, mu_0, epsilon_0
from scipy.special import iv, kv

from pmrf.core import ParametricNetwork
from pmrf.modeling.elements.distributed import RLGCLine
from pmrf.misc.inspection import get_args_of

class PhysicalCoax(RLGCLine):
    """
    A coaxial line defined directly by its physical properties (geometric and material):
        - din, inner diameter
        - dout, outer diameter
        - len, length
        - th (optional), thickness of outer-shield. If not passed, the high-frequency skin effect approximation is used.
        - epr (optional), dielectric constant (default: 'constant' vs frequency)
        - mur (optional), relative permeability (default: 'constant' vs frequency)
        - tand (optional), loss tangent (default: 'constant' vs frequency)
        - rho or <rhoin, rhoout> (optional), resistivity of conductors (default: 'constant' vs frequency)
        
    For all parameters 'x' above that are functions of frequency, several functional forms exist, specified by passing x_model in kwargs:
        - 'constant': constant function (default), parameters = ['x']
        - 'polynomialN': polynomial with order N, parameters = ['x_0', 'x_1', ..., 'x_N']
        
    Depending on the models above, the parameters for the object will change accordingly.
    """    
    def __init__(self, din = 1.12e-3, dout = 3.2e-3, len = 1.0, freq_bounds = None, neglect_skin_inductance = False, name = 'coax', **kwargs):
        self.freq_bounds = freq_bounds
        self.neglect_skin_inductance = neglect_skin_inductance
        
        self.epr_model = kwargs.get('epr_model', 'constant')
        self.mur_model = kwargs.get('mur_model', 'constant')
        self.tand_model = kwargs.get('tand_model', 'constant')
        self.rhoin_model = self.rhoout_model = self.rho_model = kwargs.get('rho_model', 'constant')
        
        params = {
            'din': din,
            'dout': dout,
            'len': len,
        }
        
        fparams = {
            'epr': 1.0,
            'mur': 1.0,
            'tand': 0.0,
            'rho': 1.68e-8,
        }
        
        if 'th' in kwargs:
            params['th'] = kwargs['th']
            self.use_hf_approx = False
        else:
            self.use_hf_approx = True
        
        if 'rhoin' in kwargs:
            fparams['rhoin'] = fparams['rhoout'] = fparams['rho']
            fparams.pop('rho')
        
        for param_name, param_default in fparams.items():
            model = getattr(self, f'{param_name}_model')
            if model == 'constant':
                params[param_name] = kwargs.get(f'{param_name}', param_default)
            elif model.startswith('polynomial'):
                k = str.__len__('polynomial')
                n = int(model[k:])
                params[f'{param_name}_0'] = kwargs.get(f'{param_name}_0', param_default)
                for i in range(1, n+1):
                    key = f'{param_name}_{i}'
                    params[key] = kwargs.get(key, 0.0)
        
        super().__init__(params=params, name=name, **kwargs)
        
    def value_f(self, param_name):
        model = str(getattr(self, f'{param_name}_model'))
        fn = self.fn
        
        if model == 'constant':
            value = getattr(self, param_name) * np.ones(self.frequency.npoints)
        elif model.startswith('polynomial'):
            n = int(model[len('polynomial'):])
            value = self.params[f'{param_name}_{0}'] * np.ones(self.frequency.npoints)
            for i in range(1, n+1):
                coeff = self.params[f'{param_name}_{i}']
                value += coeff * fn**i
                
        return value
            
    @property
    def fn(self):
        if not self.freq_bounds is None:
            f_start, f_stop = self.freq_bounds
        else:
            f_start, f_stop = self.frequency.start, self.frequency.stop            
        return (self.frequency.f - f_start) / (f_stop - f_start)            
        
    @property
    def epr_f(self):
        return self.value_f('epr')
    
    @property
    def tand_f(self):
        return self.value_f('tand')
    
    @property
    def rhoin_f(self):
        if hasattr(self, 'rhoin'):
            return self.value_f('rhoin')
        else:
            return self.value_f('rho')

    @property
    def rhoout_f(self):
        if hasattr(self, 'rhoout'):
            return self.value_f('rhoout')
        else:
            return self.value_f('rho')
    
    @property
    def mur_f(self):
        return self.value_f('mur')
    
    @property
    def eps_f(self):
        epr = self.epr_f
        tand = self.tand_f
        return epsilon_0 * epr * (1 - 1j * tand)
    
    @property
    def mu_f(self):
        mur = self.mur_f
        return mu_0 * mur
    
    @property
    def L_prime(self):
        a, b = self.din / 2, self.dout / 2
        lnbOvera = np.log(b/a)
        return self.mu_f / (2 * np.pi) * lnbOvera
    
    @property
    def C_prime(self):
        a, b = self.din / 2, self.dout / 2
        lnbOvera = np.log(b/a)
        return 2 * np.pi * np.real(self.eps_f) / lnbOvera
    
    @property
    def G_diel(self):
        a, b = self.din / 2, self.dout / 2
        lnbOvera = np.log(b/a)
        w = self.frequency.w
        return 2 * np.pi * w * -np.imag(self.eps_f) / lnbOvera
        
    @property
    def R_skin(self):
        return np.real(self.Z_skin)
    
    @property
    def L_skin(self):
        return np.imag(self.Z_skin) / self.frequency.w
    
    @property
    def Z_skin(self):
        if self.use_hf_approx:
            a, b, mu, sigma_a, sigma_b = self.din / 2, self.dout / 2, self.mu_f, 1 / self.rhoin_f, 1 / self.rhoout_f
            w = self.frequency.w
            
            L_skin_a = (1 / (2 * np.pi * a)) * np.sqrt(mu / (2 * w * sigma_a))
            L_skin_b = (1 / (2 * np.pi * b)) * np.sqrt(mu / (2 * w * sigma_b))
            L_skin = L_skin_a + L_skin_b
                    
            R_skin_a = (1 / (2 * np.pi * a)) * np.sqrt(w * mu / (2 * sigma_a))
            R_skin_b = (1 / (2 * np.pi * b)) * np.sqrt(w * mu / (2 * sigma_b))
            R_skin = R_skin_a + R_skin_b            
            
            return R_skin + 1j * self.frequency.w * L_skin
        else:
            a, b, mu, sigma_a, sigma_b = self.din / 2, self.dout / 2, self.mu_f, 1 / self.rhoin_f, 1 / self.rhoout_f
            c = b + self.th
            w = self.frequency.w

            # Za
            eta_a = np.sqrt((1j * w * mu) / sigma_a)
            gamma_a = np.sqrt(1j * w * sigma_a * mu)
            num = eta_a * iv(0, gamma_a * a)
            den = 2 * np.pi * a * iv(1, gamma_a * a)
            Za_skin = num / den
        
            # Zb
            eta_b = np.sqrt((1j * w * mu) / sigma_b)
            gamma_b = np.sqrt(1j * w * sigma_b * mu)
            num = iv(0, gamma_b * b) * kv(1, gamma_b * c) + kv(0, gamma_b * b) * iv(1, gamma_b * c)
            den = iv(1, gamma_b * c) * kv(1, gamma_b * b) - iv(1, gamma_b * b) * kv(1, gamma_b * c)
            Zb_skin = (eta_b * num) / (2*np.pi*b*den)
            
            return Za_skin + Zb_skin
    
    def compute(self):
        # All formulae from Frederick M. Tesche - 'A Simple Model for the Line Parameters of a Lossy Coaxial Cable Filled With a Nondispersive Dielectric' as well as Pozar for G
        
        L = self.L_prime
        C = self.C_prime
        G = self.G_diel
        R = np.zeros(G.shape)
        
        if not self.neglect_skin_inductance:
            L += self.L_skin
        R = self.R_skin
        
        self.R, self.L, self.G, self.C = R, L, G, C
        
        super().compute() 

class DatasheetCoax(RLGCLine):
    
    """
    A coaxial line defined by constants typically found on datasheets:
        - K1, skin effect loss (~ sqrt(w))
        - K2, dielectric loss (~ w)
        - epr, dielectric constant (1 / vf**2)
        - Zn, nominal characteristic impedance

    Additionally, the dielectric constant can be sloped (if sloped_epr=True) in which case an additional parameter epr_slope is used.
    """
    
    def __init__(self, zn = 50.0, epr = 1.0, k1 = 0.0, k2 = 0.0, len = 1.0, kc = 0.0, sloped_epr=False, slope_freq_bounds=None, separate_connector_loss=False, floating = False, loss_coeffs_normalized = False, Zc_real = False, name = 'coax', **kwargs):
        """
        epr_order indicates the order of the dielectric constant, in which case parameters epr_a1, epr_a2, epr_a3 etc. (linear, quadratic, cubic etc.)
        will be used and can be passed in kwargs.
        
        Note that freq_bounds can be passed to specify frequency bounds different to the model's bounds for e.g. the dielectric constant polynomial.

        If loss_coeffs_normalized == False:
            k1 is in units dB/100m/sqrt(MHz)
            k2 is in units dB/100m/MHz
        If loss_coeffs_normalized == True:
            k1 is in units dB/1m/sqrt(rad)
            k2 is in units dB/100m/sqrt(rad)        
        
        """
        self.floating = floating
        self.loss_coeffs_normalized = loss_coeffs_normalized
        self.use_kc = separate_connector_loss
        self.log10 = np.log(10)
        self.k1_norm_inv = 1.0 / (100 * np.sqrt(2*np.pi * 10**6)) # note that these optimizations do add up, although not much
        self.k2_norm_inv = 1.0 / (100 * 2*np.pi * 10**6)

        self.sloped_epr = sloped_epr
        self.slope_freq_bounds = slope_freq_bounds
        
        if Zc_real:
            method = 'approx'
        else:
            method = 'exact'

        params = get_args_of(float)
        if not separate_connector_loss:
            params.pop('kc')

        if self.sloped_epr:
            params['epr_slope'] = kwargs.get('epr_slope', 0.0)

        super().__init__(params=params, floating=floating, method=method, name=name, **kwargs)
        
    def compute(self):
        zn, k1, k2 = self.zn, self.k1, self.k2
        
        epr = np.ones(self.frequency.npoints) * self.epr

        frequency = self.frequency
        f = frequency.f
        
        if self.sloped_epr:
            if not self.slope_freq_bounds is None:
                f_start, f_stop = self.slope_freq_bounds
            else:
                f_start, f_stop = frequency.start, frequency.stop

            fn = (f - f_start) / (f_stop - f_start)            
            epr += self.epr_slope * fn
        
        if not self.loss_coeffs_normalized:
            k1_norm = k1 * self.k1_norm_inv
            k2_norm = k2 * self.k2_norm_inv
            if self.use_kc:
                kc_norm = self.kc * self.k1_norm_inv
        else:
            k1_norm = k1
            k2_norm = k2
            if self.use_kc:
                kc_norm = self.kc

        w = self.frequency.w
        sqrt_w = np.sqrt(w)

        dBtoNeper = self.log10 / 20
        alpha_c = k1_norm * dBtoNeper * sqrt_w
        alpha_d = k2_norm * dBtoNeper * w
        
        if self.use_kc:
            alpha_c += (kc_norm * dBtoNeper * sqrt_w) / self.len
        
        sqrt_epr = np.sqrt(epr)
        self.L = (zn * sqrt_epr) / c
        self.C = (sqrt_epr) / (zn * c)
        self.R = 2*zn * alpha_c
        self.G = 2/zn * alpha_d
        
        super().compute()        
          

class MicrostripLine(RLGCLine):
    """
    A microstrip line defined by its geometric and material properties (i.e. width, height, dielectric constant, tan delta, rho).
    """
    def __init__(self, W = 3e-3, H = 1.6e-3, len = 1.0, epr = 4.3, tand = 0.0, rho = 0.0, floating = False, loss_coeffs_normalized = False, Zc_real = False, name = 'mstrip', **kwargs):
        self._floating = floating
        self._loss_coeffs_normalized = loss_coeffs_normalized
        if Zc_real:
            method = 'approx'
        else:
            method = 'exact'
            
        super().__init__(params=get_args_of(float), floating=floating, method=method, name=name, **kwargs)    
        
    def compute(self):
        W, H = self.W, self.H
        epr, tand, rho = self.epr, self.tand, self.rho
        omega = self.frequency.w
        
        u = W / H
        if u < 1.0:
            raise Exception('h > w not supported yet')

        t1 = ((epr + 1) / 2)
        t2 = ((epr - 1) / 2)
        t3 = 1 / np.sqrt(1 + 12 / u)
        epe = (t1 + t2*t3) * np.ones(self.frequency.npoints)
        
        Za = (120 * np.pi) / (u + 1.393 + 0.667 * np.log(u + 1.444))
        Ze = Za / np.sqrt(epe)

        self.L = (Ze * np.sqrt(epe)) / c
        self.C = (np.sqrt(epe)) / (Ze * c)
        self.R = (1 / W) * np.sqrt(2 * mu_0 * rho) * np.sqrt(omega)
        self.G = (1 / (Za * c)) * (epr * (epe - 1) / (epr - 1)) * tand * omega
        
        super().compute()

class SteppedCoaxialLine(ParametricNetwork):
    def __init__(self, zn_init = 50.0, epr_init = 1.0, k1 = 0.0, k2 = 0.0, len = 1.0, N = 2, name = 'coax', **kwargs):
        self.N = N
        params = {}

        for i in range(N):
            params[f'zn{i}'] = zn_init
            params[f'epr{i}'] = epr_init

        params['k1'] = k1
        params['k2'] = k2
        params['len'] = len

        super().__init__(params, nports=2, name=name, **kwargs)

    def compute(self):
        # Create a list of CoaxialLine objects
        lines = []
        for i in range(self.N):
            zn = self.params[f'zn{i}']
            epr = self.params[f'epr{i}']
            line = DatasheetCoax(zn=zn, epr=epr, k1=self.k1, k2=self.k2, len=self.len/self.N, frequency=self.frequency)
            lines.append(line)

        # Connect the lines in series
        cascade = lines[0]
        for n in lines[1:]:
            cascade = cascade ** n

        self.s = cascade.s
        

class SmoothCoaxialLine(ParametricNetwork):
    def __init__(self, zn = 50.0, epr = 1.0, k1 = 0.0, k2 = 0.0, len = 1.0, num_vertices = 3, num_lines = 50, model = 'spline', name = 'coax', **kwargs):
        self.num_vertices = num_vertices
        self.num_lines = num_lines
        self.model = model
        params = {}
        
        if num_vertices < 2:
            raise Exception('Must have at least two end vertices')

        for i in range(num_vertices):
            params[f'zn{i}'] = zn
            params[f'epr{i}'] = epr

        params['k1'] = k1
        params['k2'] = k2
        params['len'] = len

        super().__init__(params, nports=2, name=name, **kwargs)
    
    def compute(self):
        # Get the zn and epr functional objects
        zn_f = self.zn_f
        epr_f = self.epr_f
        
        # Create a list of CoaxialLine objects
        x_vals = np.linspace(0, self.len, self.num_lines)
        for x in x_vals:
            zn = zn_f(x)
            epr = epr_f(x)
            a_new = self.get_abcd(zn, epr, self.k1, self.k2, self.len/self.num_lines)
            if x == 0.0:
                a = a_new
            else:
                a = a @ a_new
        self.a = a
        
    def get_abcd(self, zn, epr, k1, k2, len):
        k1_norm = k1 / (100 * np.sqrt(2*np.pi * 10**6))
        k2_norm = k2 / (100 * 2*np.pi * 10**6)

        w = self.frequency.w

        epr = epr * np.ones(self.frequency.npoints)
        
        dBtoNeper = np.log(10) / 20
        alpha_c = k1_norm * dBtoNeper * np.sqrt(w)
        alpha_d = k2_norm * dBtoNeper * w

        L = (zn * np.sqrt(epr)) / c
        C = (np.sqrt(epr)) / (zn * c)
        R = 2*zn * alpha_c
        G = 2/zn * alpha_d
        
        gamma = np.sqrt((R + 1j*w*L) * (G + 1j*w*C))
        z0 = np.sqrt((R + 1j*w*L) / (G + 1j*w*C))
        
        a = np.zeros((self.frequency.npoints, 2, 2), dtype=complex)
        a[:, 0, 0] = np.cosh(gamma * len)
        a[:, 0, 1] = z0 * np.sinh(gamma * len)
        a[:, 1, 0] = (1/z0) * np.sinh(gamma * len)
        a[:, 1, 1] = np.cosh(gamma * len)
        return a
        
    # Return the characteristic impedance functional object
    @property
    def zn_f(self):
        y = np.array([self.params[f'zn{i}'] for i in range(self.num_vertices)])
        x = np.linspace(0, self.len, self.num_vertices)        
        return CubicSpline(x, y)
    
    # Return the dielectric constant functional object
    @property
    def epr_f(self):
        y = np.array([self.params[f'epr{i}'] for i in range(self.num_vertices)])
        x = np.linspace(0, self.len, self.num_vertices)        
        return CubicSpline(x, y)
    
    
class SlopedLine(ParametricNetwork):
    def __init__(self, zn = 50.0, eps_r_const = 1.0, eps_r_slope=0.0, tan_d_const=0.0, tan_d_slope=0.0, r_prime_const = 0.0, r_prime_slope = 0.0, len = 1.0, slope_freq_bounds=None, **kwargs):
        self.slope_freq_bounds = slope_freq_bounds
               
        super().__init__(params=get_args_of(float), nports=2, **kwargs)
    
    def compute(self):
        eps_r_const, eps_r_slope = self.eps_r_const, self.eps_r_slope
        tan_d_const, tan_d_slope = self.tan_d_const, self.tan_d_slope
        r_prime_const, r_prime_slope = self.r_prime_const, self.r_prime_slope
        zn, len = self.zn, self.len
        
        f = self.frequency.f
    
        frequency = self.frequency
        
        if not self.slope_freq_bounds is None:
            f_start, f_stop = self.slope_freq_bounds
        else:
            f_start, f_stop = frequency.start, frequency.stop
        
        fn = (f - f_start) / (f_stop - f_start)            

        eps_r = eps_r_slope * fn + eps_r_const
        tan_d = tan_d_slope * fn + tan_d_const
        r_prime = r_prime_slope * fn + r_prime_const

        sqrt_eps_r = np.sqrt(eps_r)
        alpha_c = r_prime / (2 * zn)
        alpha_d = np.pi * sqrt_eps_r * tan_d * f / c

        beta = 2*np.pi * f / c * sqrt_eps_r
        
        gamma = alpha_c + alpha_d + 1j * beta

        gL = gamma * len
        sinh_gL, cosh_gL = np.sinh(gL), np.cosh(gL)
        a = np.zeros((self.frequency.npoints, 2, 2), dtype=complex)
        a[:, 0, 0] = cosh_gL
        a[:, 0, 1] = zn * sinh_gL
        a[:, 1, 0] = 1 / zn * sinh_gL
        a[:, 1, 1] = a[:, 0, 0]

        self.a = a

    def subdivide(self, n=10):
        lines = []
        for i in range(n):
            line = self.copy()
            line.len = self.len / n
            line.name = f'{self.name}_{i+1}'
            lines.append(line)

        return lines


class BasicLine(ParametricNetwork):
    def __init__(self, zn = 50.0, eps_r = 1.0, tan_d=0.0, r_prime = 0.0, len = 1.0, **kwargs):
        super().__init__(params=get_args_of(float), nports=2, **kwargs)
    
    def compute(self):
        eps_r, tan_d, r_prime = self.eps_r, self.tan_d, self.r_prime
        zn, len = self.zn, self.len
        
        frequency = self.frequency
        f = frequency.f

        alpha_c = r_prime / (2 * zn)
        alpha_d = np.pi * np.sqrt(eps_r) * tan_d * f / c

        beta = 2*np.pi * f / c * np.sqrt(eps_r)
        
        gamma = alpha_c + alpha_d + 1j * beta

        gL = gamma * len
        a = np.zeros((self.frequency.npoints, 2, 2), dtype=complex)
        a[:, 0, 0] = np.cosh(gL)
        a[:, 0, 1] = zn * np.sinh(gL)
        a[:, 1, 0] = 1 / zn * np.sinh(gL)
        a[:, 1, 1] = a[:, 0, 0]

        self.a = a    