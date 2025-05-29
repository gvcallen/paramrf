import numpy as np
import skrf as rf

def add_noise(network: rf.Network, sigma_gamma=0.1, sigma_tau=None):
    num_freq = network.frequency.npoints
    
    if sigma_tau is None:
        sigma_tau = sigma_gamma
    
    for m, n in network.port_tuples:
        if m == n:
            sigma = sigma_gamma
        else:
            sigma = sigma_tau
        noise = sigma * np.random.randn(num_freq) + 1j*sigma * np.random.randn(num_freq)                

        network.s[:, m, n] += noise   