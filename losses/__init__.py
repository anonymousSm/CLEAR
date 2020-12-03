from __future__ import print_function, absolute_import

from .SoftmaxNeigLoss import SoftmaxNeigLoss
from .KNNSoftmax import KNNSoftmax
from .NeighbourLoss import NeighbourLoss
from .triplet import TripletLoss
from .NeighbourLoss import NeighbourLoss
from .BinDevianceLoss import BinDevianceLoss
from .ms_loss import MultiSimilarityLoss


__factory = {
    'softneig': SoftmaxNeigLoss,
    'knnsoftmax': KNNSoftmax,
    'neighbour': NeighbourLoss,
    'triplet': TripletLoss,
    'MSLoss': MultiSimilarityLoss,
    
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name](*args, **kwargs)


