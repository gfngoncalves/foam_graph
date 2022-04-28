class KeepOnlyMainAttrs(object):
    r"""Removes all attributes except the desired ones.

    Args:
        attr (list): List of attributes to keep.
    """

    def __init__(self, attrs=("x", "edge_index", "edge_attr", "y", "pos")):
        self.attrs = attrs

    def __call__(self, data):
        for attr in data.to_dict():
            if attr not in self.attrs:
                delattr(data, attr)
        return data

    def __repr__(self):
        return "{}(attrs={})".format(self.__class__.__name__, self.attrs)
