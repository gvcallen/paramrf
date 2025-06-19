# from typing import Callable, Any, Dict, get_args, get_origin, Union
# from types import UnionType

# from pmrf._model import Model
# from pmrf._misc import field

# class CompoundModel(Model):
#     _submodels: list = field(default_factory=[], init=False, static=True)

#     def __init_subclass__(cls, dynamic = None, **kwargs):
#         super().__init_subclass__(dynamic, **kwargs)
#         for field_name, field_types in cls.__annotations__.items():
#             # The annotations could be unions - in this case we just take the first one TODO upgrade this to do more in-depth inspection?
#             origin = get_origin(field_types)
#             if origin in (Union, UnionType):
#                 field_type = get_args(field_types)[0]
#             else:
#                 field_type = field_types
#         print('')

#     @property
#     def models(self) -> list[Model]:
#         # TODO implement this automagically
#         raise NotImplementedError("'models' property must be implemented sub-classes for a CompoundModel")
    
#     @property
#     def num_submodels(self):
#         return len(self.models)