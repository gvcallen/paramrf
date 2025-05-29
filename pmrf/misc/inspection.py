import inspect
import sys

def get_args_of(type_to_fetch=float, depth=1) -> dict:
    """
    Magic function to get parameters in the calling scope of a certain type, and store them in a dictionary,
    where the name is the parameter name, and the value is the value within the scope.
    """
    depth = sys._getframe(depth)
    func_name = depth.f_code.co_name
    
    instance = depth.f_locals['self']
    
    # Thanks ChatGPT
    for cls in type(instance).mro():
        if func_name in cls.__dict__ and cls.__dict__[func_name].__code__ == depth.f_code:
            func_handle = getattr(cls, func_name)
            break
    
    signature = inspect.signature(func_handle)
    local_variables = depth.f_locals
    
    # Filter parameters with default values that are floats
    params = {}
    for param in signature.parameters.values():
        try:
            try_use = False
            default, annotation = param.default, param.annotation
            if default is not inspect._empty:
                if isinstance(default, type_to_fetch):
                    try_use = True
            if annotation is not inspect._empty:
                if issubclass(annotation, type_to_fetch):
                    try_use = True

            if try_use:
                value = local_variables[param.name]
                if try_use and not value is None and not param.name == 'self':
                    params[param.name] = value
        except:
            pass
        
    return params    

def get_param_values(type_to_fetch=float) -> list:
    return list(get_args_of(type_to_fetch, depth=2).values())

def get_param_names(type_to_fetch=float) -> list:
    return list(get_args_of(type_to_fetch, depth=2).keys())

def get_properties_and_attributes(cls):
    return [
        name for name, value in inspect.getmembers(cls)
        if isinstance(value, property) or not callable(value) and not name.startswith('__')
    ]