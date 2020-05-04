from .blob import Blob, Digit
from .product import Product
from .timeserie import TSDataset, TimeSerie
from .derivation import Degrader
from .export import ProductDataset
from .modules import samplers

__all__ = ['Blob', 'Digit', 'Product', 'TSDataset', 'TimeSerie',
           'ProductDataset', 'Degrader', 'samplers']


def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module


class Registry(dict):
    """
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.
    Eg. creating a registry:
        some_registry = Registry({"default": default_module})
    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_module_nickname")
        def foo():
            ...
    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_module"]
    """

    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name, cls=None):
        # used as function call
        if cls is not None:
            _register_generic(self, module_name, cls.build)
            return

        # used as decorator
        def register_fn(cls):
            _register_generic(self, module_name, cls.build)
            return cls

        return register_fn

    def register_fn(self, module_name, fn=None):
        # used as function call
        if fn is not None:
            _register_generic(self, module_name, fn)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn
