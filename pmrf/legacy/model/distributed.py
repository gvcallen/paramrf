# import numpy as np
# import pmrf._core.n
# import skrf as rf
# import numpy as np
# from skrf.media import DefinedGammaZ0, DistributedCircuit
# from skrf.constants import c
# from skrf import mu_0

# from pmrf import Network, Parameter, ParameterArray

# class RLGCLine(Network):
#     R: ParameterArray
#     L: ParameterArray
#     G: ParameterArray
#     C: ParameterArray
#     len: Parameter

#     def a(self, w):
#         R, L, G, C, len = self.R, self.L, self.G, self.C, self.len
#         gamma = np.sqrt((R + 1j*w*L) * (G + 1j*w*C))
#         Zc = np.sqrt((R + 1j*w*L) / (G + 1j*w*C))

#         gL = gamma*self.len
        
#         a = np.array([
#             [np.cosh(gL), Zc * np.sinh(gL)],
#             [1 / Zc * np.sinh(gL), np.cosh(gL)]
#         ]).transpose(2, 0, 1)

#         return a
        

# class RLGCLineOld(ParametricNetwork):
#     def __init__(self, params = None, R = 0.0, L = 280e-9, G = 0.0, C = 90e-12, len = 1.0, floating = False, method='exact', **kwargs):
#         self.method = method
#         self.a_cached = None
#         self._dont_calculate_s = False
        
#         nports = 2 if not floating else 4
#         params = params if params is not None else get_args_of(float)
#         super().__init__(params=params, nports=nports, **kwargs)
        
#         if nports == 2:
#             self.a_cached = self.a
#         else:
#             self.a_cached = None
        
#     @property
#     def dont_calculate_s(self):
#         return self._dont_calculate_s
    
#     @dont_calculate_s.setter
#     def dont_calculate_s(self, value):
#         self._dont_calculate_s = value
#         self.update()
        
#     def compute(self):
#         R, L, G, C = self.R, self.L, self.G, self.C
        
#         w = self.frequency.w
#         if self.method == 'skrf':
#             media = DistributedCircuit(self.frequency, z0_port=self.z0_port, C=C, L=L, R=R, G=G)
#         else:
#             if self.method == 'exact':
#                 gamma = np.sqrt((R + 1j*w*L) * (G + 1j*w*C))
#                 Zc = np.sqrt((R + 1j*w*L) / (G + 1j*w*C))
#             elif self.method == 'semiapprox':
#                 Zn = np.sqrt(L/C)
                
#                 beta = w * np.sqrt(L*C)
#                 alpha_c = R / (2*Zn)
#                 alpha_d = (G*Zn)/2

#                 Zc = Zn
#                 gamma = 1j*beta + alpha_c + alpha_d                
#             elif self.method == 'approx':
#                 Zn = np.sqrt(L/C)
                
#                 beta = w * np.sqrt(L*C)
#                 alpha_c = R / (2*Zn)
#                 alpha_d = (G*Zn)/2

#                 gamma = 1j*beta + alpha_c + alpha_d
#                 Zc = Zn
                        
#         if self.number_of_ports == 2:
#             a = np.zeros((self.frequency.npoints, 2, 2), dtype=complex)
#             gL = gamma*self.len
#             a[:, 0, 0] = np.cosh(gL)
#             a[:, 0, 1] = Zc * np.sinh(gL)
#             a[:, 1, 0] = 1 / Zc * np.sinh(gL)
#             a[:, 1, 1] = a[:, 0, 0]
            
#             self.a_cached = a
#             if not self._dont_calculate_s:
#                 self.s = rf.network.a2s(a, self.z0_port)
#         else:
#             # NB TODO: This method using DefinedGammaZ0 is slowwww! Find alternative
#             media = DefinedGammaZ0(frequency=self.frequency, z0_port=self.z0_port, z0=Zc, gamma=gamma)
#             self.s = media.line_floating(self.len, 'm').s