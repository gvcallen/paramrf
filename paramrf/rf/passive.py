import numpy as np
import skrf as rf

# The available gain of a two-port network relative to the available power of a source, where port 1 of the two-port is connected to the source
def available_gain(source: rf.Network, passive: rf.Network, source_port=0):
    if not source_port == 0:
        passive = passive.flipped()
    
    # Note: Equation 10.25 on page 507 of Pozar, using the references variable names from Pozar
    ntwk_back = passive.flipped() ** source
    
    gamma_S = source.s[:, 0, 0]

    gamma_out = ntwk_back.s[:, 0, 0]
    s21, s11 = passive.s[:, 1, 0], passive.s[:, 0, 0]
    
    num = np.abs(s21)**2 * (1 - np.abs(gamma_S)**2)
    den = np.abs(1 - s11*gamma_S)**2 * (1 - np.abs(gamma_out)**2)
    Gav = num / den

    return Gav

def effective_temperature(source: rf.Network, passive: rf.Network, Tp):
    Gav = available_gain(source, passive)
    return ((1 - Gav) / Gav) * Tp

def output_temperature(source: rf.Network, passive: rf.Network, Ts, Tp):
    Gav = available_gain(source, passive)
    return Ts * Gav + (1 - Gav) * Tp

def output_temperature_alternative(source: rf.Network, passive: rf.Network, Ts, Tp):   
    # Calculate gamma looking back towards generator
    gamma_2 = rf.connect(source, 0, passive, 0).s[:,0,0]
        
    # We want to calculate an effective output temperature (independent of the load) so we assume the load is a conjugate match i.e. gamma_L = gamma_2*
    load = rf.Network(frequency=source.frequency, s=np.conj(gamma_2), z0=passive.z0[:,1])
    gamma_L = load.s[:,0,0]
    
    # Calculate other gammas
    gamma_1 = rf.connect(passive, 1, load, 0).s[:,0,0]
    gamma_G = source.s[:,0,0]
    gamma_3 = gamma_L
    
    tau_1 = np.abs(passive.s[:,1,0])**2
    s22 = passive.s[:,1,1]
    
    num = (1 - np.abs(gamma_G)**2) * tau_1
    den = np.abs(1 - gamma_G * gamma_1)**2 * np.abs(1 - s22*gamma_3)**2
    tau_m1 = num / den
    tau_M1 = tau_m1 * (1 - np.abs(gamma_L)**2)
    
    I1p = (1 - np.abs(gamma_2)**2) / (np.abs(1 - gamma_2 * gamma_3))**2
    I1 = I1p * (1 - np.abs(gamma_L)**2)
    
    T2d = Ts*tau_M1 + Tp*(I1 - tau_M1)
    return T2d


def cascaded_output_temperature(source: rf.Network, passives: list[rf.Network], Ts, temperatures):
    # gamma_backwards[n] is the backward reflection coefficient just after network n looking towards the generator
    gamma_G = source.s[:,0,0]
    gamma_backwards = [gamma_G]
    source_cascaded = source
    for passive in passives:
        source_cascaded = rf.connect(source_cascaded, 0, passive, 0)
        gamma_backwards.append(source_cascaded.s[:,0,0])
        
    # We want to calculate an effective output temperature (independent of the load) so we assume the load is a conjugate match i.e. gamma_L = gamma_back*
    load = rf.Network(frequency=source.frequency, s=np.conj(gamma_backwards[-1]), z0=passive.z0[:,1])
    gamma_L = load.s[:,0,0]
        
    # gamma_forwards[n-1] is the forward reflection coefficient just before network n towards the load
    gamma_forwards = [gamma_L]
    load_cascaded = load
    for passive in reversed(passives):
        load_cascaded = rf.connect(passive, 1, load_cascaded, 0)
        gamma_forwards.insert(0, load_cascaded.s[:,0,0])

    # Combine networks so that i=0 corresponds to the load, i=1 corresponds to the first two-port, and i=N+1 corresponds to the load
    networks = [source] + passives + [load]
    N = len(passives)
    
    def tau(k):
        s21 = networks[k].s[:,1,0]
        return np.abs(s21)**2
    
    def gamma(k):
        if k % 2 == 0:
            i = int((k + 2) / 2)
            n = i - 1
            return gamma_backwards[n]
        else:
            # Odd subscript
            i = int((k - 1) / 2 )   
            n = i + 1
            return gamma_forwards[n-1]

    def tau_m(i):
        gamma_2iMinus2 = gamma(2*i - 2)
        gamma_2iMinus1 = gamma(2*i - 1)
        num = 1 - np.abs(gamma_2iMinus2)**2
        den = np.abs(1 - gamma_2iMinus2 * gamma_2iMinus1)**2
        coeff = num / den

        product = 1
        for j in range(i, N+1):
            s22 = networks[j].s[:,1,1]
            num = tau(j)
            den = np.abs(1 - s22 * gamma(2*j + 1))**2
            product *= num / den

        return coeff * product
    
    def tau_M(i):
        return tau_m(i) * (1 - np.abs(gamma_L)**2)
    
    def M(i):
        return tau_m(i+1)
    
    def M_id(i):
        return M(i) * (1 - np.abs(gamma_L)**2)
    
    T_Td = Ts * tau_M(1)
    for i in range(1, N+1):
        T_Td += temperatures[i-1] * (M_id(i) - tau_M(i))

    return T_Td