"""
Adapter models that wrap Python collections.
"""

from pmrf.models import Model
    
class ListModel(Model):
    """
    A container model that holds a list of sub-models.

    Parameters
    ----------
    models : list[Model]
        The list of child models.
    """
    #: The models.
    models: list[Model]


class DictModel(Model):
    """
    A container model that holds a dictionary of sub-models.

    Parameters
    ----------
    models : dict[str, Model]
        A dictionary mapping names to child models.
    """
    #: The models.
    models: dict[str, Model]

    def __post_init__(self):
        """
        Automatically sets the dictionary items as attributes of the instance.
        """
        for key, value in self.models:
            setattr(self, key, value)