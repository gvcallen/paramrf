import pmrf as prf
from pmrf.models import Resistor
from pmrf.parameters import Uniform
from pmrf.sampling import LatinHypercubeSampler

resistor = Resistor(R=Uniform(9.0, 11.0))
sampler = LatinHypercubeSampler(resistor)
resistors = sampler._generate_models(10)
freq = prf.Frequency(10, 20, 100, 'MHz')

for i, res in enumerate(resistors):
    res.export_touchstone(freq, f'resistors_{i}')