import torch


class Make2D(object):
    r"""Scales an attribute to a given range.

    Args:
        attr (str): The attribute to stack.
        fields (list of lists or list of str): Fields and components to stack.
    """
    plane_to_comps = {"xy": (0, 1), "xz": (0, 3), "yz": (2, 3)}

    def __init__(self, plane):
        self.plane = plane
        if self.plane not in self.plane_to_comps:
            raise ValueError("invalid plane specification")
        self.comps = self.plane_to_comps[self.plane]

    def __call__(self, data):
        data.pos = data.pos[:, self.comps]
        return data

    def __repr__(self):
        return "{}(plane={}, comps={})".format(
            self.__class__.__name__, self.plane, self.comps
        )
