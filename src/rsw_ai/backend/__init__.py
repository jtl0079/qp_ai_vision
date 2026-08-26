# src/rsw_ai/backend/__init__.py

# __init__.py

import pkgutil
from importlib import import_module

for _, name, _ in pkgutil.iter_modules(__path__):
    globals()[name] = import_module(f"{__name__}.{name}")
