import torch


class Stack(object):
    r"""Creates an attribute by stacking other attributes.

    Args:
        attr (str): The attribute to stack.
        fields (list of lists or list of str): Fields and components to stack.
    """

    def __init__(self, attr, fields):
        self.attr = attr
        self.fields = fields

    def __call__(self, data):
        data[self.attr] = torch.hstack(
            [
                data[f]
                if isinstance(f, str) or (data[f[0]].size(-1)) == 1
                else data[f[0]][:, f[1]]
                for f in self.fields
            ]
        )
        return data

    def unstack(self, data):
        i = 0
        for field in self.fields:
            # TODO: improve unstacking
            c = 1 if isinstance(field, str) else len(field[1])
            f = field if isinstance(field, str) else field[0]
            data[f] = data[self.attr][:, i : i + c]
            i += c
        return data

    def __repr__(self):
        return "{}(attr={}, fields={})".format(
            self.__class__.__name__, self.attr, self.fields
        )
