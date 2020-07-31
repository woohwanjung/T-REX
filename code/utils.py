import numpy as np

class dict_set(dict):
    def __init__(self):
        super().__init__()

    def add(self, key, val):
        if key not in self:
            self[key] = set()
        self[key].add(val)







def descobj(obj):
    if isinstance(obj, dict):
        for k in obj.keys():
            print(k)
            desc(obj[k], True)
    else:
        for k in obj.__dict__.keys():
            print(k)
            desc(getattr(obj, k), True)


def desc(obj, short=False):
    if isinstance(obj, list):
        print("List", f"Len: {len(obj)}")
    elif isinstance(obj, dict):
        print("Dict", f"Len: {len(obj)}")
    elif isinstance(obj, np.ndarray):
        print(f"numpy ndarray: {obj.shape}")
    elif isinstance(obj, int):
        print("Int", obj)
    elif isinstance(obj, str):
        print("Str", obj)
    else:
        if short:
            print(type(obj))
        else:
            help(obj)
            dir(obj)