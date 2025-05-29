import numpy as np
import skrf as rf

def get_twoport_a(network: rf.Network, a_buffer=None):
    # From Pozar parameter conversion table. Ever slightly faster than the general scikit-rf method
    a = a_buffer if a_buffer is not None else np.zeros((network.s.shape), dtype=np.complex128)
    z0 = network.z0[:,0]
    s11, s12, s21, s22 = network.s[:,0,0], network.s[:,0,1], network.s[:,1,0], network.s[:,1,1]
    inv = 1 / (2.0*s21)
    a[:,0,0] = ((1+s11)*(1-s22) + s12*s21) * inv
    a[:,0,1] = ((1+s11)*(1+s22) - s12*s21) * inv * z0
    a[:,1,0] = ((1-s11)*(1-s22) - s12*s21) * inv / z0
    a[:,1,1] = ((1 - s11) * (1 + s22) + s12*s21) * inv
    return a

def concatenate_twoports(networks: list[rf.Network], return_a=False):
    # a = networks[0].a
    a = get_twoport_a(networks[0])
    a_buffer = None
    for network in networks[1:]:
        if network is None:
            continue
        # a = a @ network.a
        a_buffer = get_twoport_a(network, a_buffer=a_buffer)
        a = a @ a_buffer

    if return_a:
        return a
    else:
        rf.Network(frequency=networks[0].frequency, a=a, z0=networks[0].z0)

def terminate_a_in_s11(a, s11, z0):
    A, B, C, D = a[:,0,0], a[:,0,1], a[:,1,0], a[:,1,1]
    num = z0 * (1 + s11) * (A - z0*C) + (B - D*z0)*(1-s11)
    den = z0 * (1 + s11) * (A + z0*C) + (B + D*z0)*(1-s11)
    return num / den