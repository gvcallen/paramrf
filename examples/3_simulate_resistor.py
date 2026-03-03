import pmrf as prf
from pmrf.models import Resistor
from pmrf.parameters import Uniform
from pmrf.sampling import LatinHypercubeSampler

resistor = Resistor(R=Uniform(9.0, 11.0))
sampler = LatinHypercubeSampler(resistor)
results = sampler.run(N=10)
freq = prf.Frequency(10, 20, 100, 'MHz')

for i, params in enumerate(results.sampled_params):
    resistor.with_params(params).export_touchstone(f"resistor_{i}", freq)