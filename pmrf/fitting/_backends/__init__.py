import importlib

backends = ['anesthetic', 'blackjax', 'dypolychord', 'numpyro', 'optax', 'polychord', 'scipy']

for backend in backends:
    try:
        module = importlib.import_module(f"fitting._backends.{backend}")
        globals().update({k: v for k, v in module.__dict__.items() if not k.startswith('_')})
    except ImportError:
        pass