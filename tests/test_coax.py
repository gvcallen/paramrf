from pmrf.models.lines import DatasheetCoaxial, PhysicalCoaxial
from pmrf.model_legacy.lines import DatasheetCoax as DatasheetCoaxOld
from pmrf.model_legacy.lines import PhysicalCoax as PhysicalCoaxOld
import skrf as rf

import jax
import jax.numpy as jnp

import inspect

frequency = rf.Frequency(50, 200, 151, 'MHz')
coax = DatasheetCoaxial(length=10, zn=51)

coax_old = DatasheetCoaxOld(len=10, zn=51, frequency=frequency)
coax.to_skrf(frequency).plot_s_db(m=0, n=0)
coax_old.plot_s_db(m=0, n=0)