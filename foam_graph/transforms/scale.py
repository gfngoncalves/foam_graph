class Scale(object):
    r"""Scales an attribute to a given range.

    Args:
        attr (str, optional): The attribute to scale. (default: :obj:`x`)
        attr_min (float or Tensor, optional): Minimum values. (default: :obj:`0`)
        attr_max (float or Tensor, optional): Maximum values. (default: :obj:`1`)
    """

    def __init__(self, attr="x", attr_min=0, attr_max=1):
        self.attr = attr
        self.attr_min = attr_min
        self.attr_max = attr_max

    def __call__(self, data):
        data[self.attr] -= self.attr_min
        data[self.attr] /= self.attr_max - self.attr_min

        return data

    def unscale(self, data):
        data[self.attr] *= self.attr_max - self.attr_min
        data[self.attr] += self.attr_min

        return data

    def __repr__(self):
        return "{}(attr={}, min={}, max={})".format(
            self.__class__.__name__, self.attr, self.attr_min, self.attr_max
        )
