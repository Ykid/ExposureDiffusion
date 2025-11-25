from .ELD_model import ELDModel
from .ELD_model_iter import ELDModelIter
from .ELD_model_iter_v2 import ELDModelIterV2

def eld_model():
    return ELDModel()

def eld_iter_model():
    return ELDModelIter()

def eld_iter_v2_model():
    return ELDModelIterV2()