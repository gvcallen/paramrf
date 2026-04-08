"""
Adapter models that wrap Python collections.
"""

from pmrf.core import Model
    
class ListModel(Model):
    """
    A container model that holds a list of sub-models.

    Attributes
    ----------
    models : list[Model]
        The list of child models.
    """
    models: list[Model]


class DictModel(Model):
    """
    A container model that holds a dictionary of sub-models.

    Attributes
    ----------
    models : dict[str, Model]
        A dictionary mapping names to child models.
    """
    models: dict[str, Model]

    def __post_init__(self):
        """
        Automatically sets the dictionary items as attributes of the instance.
        """
        for key, value in self.models:
            setattr(self, key, value)