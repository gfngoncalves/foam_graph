import torch

class NormalizeZScore(torch.nn.Module):
    r"""Scales an attribute to a given mean and std.

    Args:
        attr (str, optional): The attribute to scale. (default: :obj:`x`)
        attr_mean (float or Tensor, optional): Mean values. (default: :obj:`0`)
        attr_std (float or Tensor, optional): Std. dev. values. (default: :obj:`1`)
    """

    def __init__(self, attr="x", attr_mean=torch.tensor(0.0), attr_std=torch.tensor(1.0)):
        super().__init__()
        self.attr = attr
        self.register_buffer("attr_mean", attr_mean)
        self.register_buffer("attr_std", attr_std)

    def __call__(self, data):
        data[self.attr] -= self.attr_mean
        data[self.attr] /= self.attr_std

        return data

    def unscale(self, data):
        data[self.attr] *= self.attr_std
        data[self.attr] += self.attr_mean

        return data

    def __repr__(self):
        return "{}(attr={}, mean={}, std={})".format(
            self.__class__.__name__, self.attr, self.attr_mean, self.attr_std
        )
