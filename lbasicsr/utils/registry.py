class Registry():
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    """
    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj, suffix=None):
        if isinstance(suffix, str):
            name = name + '_' + suffix

        assert (name not in self._obj_map), (f"An object named '{name}' was already registered "
                                             f"in '{self._name}' registry!")
        self._obj_map[name] = obj

    # =============================================================
    # register() 函数主要用来注册一个实现的类或函数
    def register(self, obj=None, suffix=None):
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class, suffix)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj, suffix)
    # =============================================================

    # =======================================================================================
    # get() 函数主要用来根据配置文件中的类名或函数名来查找对应的实例
    def get(self, name, suffix='lbasicsr'):
        ret = self._obj_map.get(name)
        if ret is None:
            ret = self._obj_map.get(name + '_' + suffix)
            print(f'Name {name} is not found, use name: {name}_{suffix}!')
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret
    # =======================================================================================

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()


DATASET_REGISTRY = Registry('dataset')
ARCH_REGISTRY = Registry('arch')
MODEL_REGISTRY = Registry('model')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')
