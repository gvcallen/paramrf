"""
Core maths functions.
"""

from pmrf.math.bessel import (
    i0_over_i1 as i0_over_i1,
)

import pmrf.math.aggregations as aggregations
from pmrf.math.aggregations import (
    weighted_sum as weighted_sum,
    geometric_mean as geometric_mean,
    convolution_aggregate as convolution_aggregate,
    aggregate as aggregate,
)

from pmrf.math.conversions import (
    complex_2_magnitude as complex_2_magnitude,
    complex_2_db as complex_2_db,
    complex_2_db10 as complex_2_db10,
    complex_2_radian as complex_2_radian,
    complex_2_degree as complex_2_degree,
    complex_2_quadrature as complex_2_quadrature,
    complex_2_reim as complex_2_reim,
    complex_components as complex_components,
    magnitude_2_db as magnitude_2_db,
    mag_2_db as mag_2_db,    
    mag_2_db10 as mag_2_db10,
    db_2_magnitude as db_2_magnitude,
    db_2_mag as db_2_mag,
    db10_2_mag as db10_2_mag,
    magdeg_2_reim as magdeg_2_reim,
    dbdeg_2_reim as dbdeg_2_reim,
    db_2_np as db_2_np,
    np_2_db as np_2_db,
    radian_2_degree as radian_2_degree,
    degree_2_radian as degree_2_radian,
    feet_2_meter as feet_2_meter,
    meter_2_feet as meter_2_feet,
    db_per_100feet_2_db_per_100meter as db_per_100feet_2_db_per_100meter,
    polar_2_rect as polar_2_rect,
    rect_2_polar as rect_2_polar,
    sqrt_phase_unwrap as sqrt_phase_unwrap,
    complexify as complexify,
    CONVERSION_LOOKUP as CONVERSION_LOOKUP,
)
import pmrf.math.losses as losses
from pmrf.math.losses import (
    mean_squared_error as mean_squared_error,
    log_mean_squared_error as log_mean_squared_error,
    root_mean_squared_error as root_mean_squared_error,
    mean_absolute_error as mean_absolute_error,
    mean_absolute_percentage_error as mean_absolute_percentage_error,
    huber_loss as huber_loss,
    hinge_loss as hinge_loss,
    LOSS_LOOKUP as LOSS_LOOKUP,
)
from pmrf.math.misc import *

__all__ = [
    "aggregations",
    "i0_over_i1",
    "complex_2_magnitude",
    "complex_2_db",
    "complex_2_db10",
    "complex_2_radian",
    "complex_2_degree",
    "complex_2_quadrature",
    "complex_2_reim",
    "complex_components",
    "magnitude_2_db",
    "mag_2_db",
    "mag_2_db10",
    "db_2_magnitude",
    "db_2_mag",
    "db10_2_mag",
    "magdeg_2_reim",
    "dbdeg_2_reim",
    "db_2_np",
    "np_2_db",
    "radian_2_degree",
    "degree_2_radian",
    "feet_2_meter",
    "meter_2_feet",
    "db_per_100feet_2_db_per_100meter",
    "polar_2_rect",
    "rect_2_polar",
    "sqrt_phase_unwrap",
    "complexify",
    "losses",
    "CONVERSION_LOOKUP",
    "LOSS_LOOKUP",
]