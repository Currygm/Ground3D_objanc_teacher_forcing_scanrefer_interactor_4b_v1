"""Restricted official archive reader; no MMEngine import or arbitrary pickle globals.
Callers must verify the complete official SHA256 before initializing a model.
Metadata-only preflight may use a sparse archive, never as model initialization.
"""
import codecs,collections,pickle,types
import numpy as np
import torch

class ConfigDict(dict):pass
class HistoryBuffer:
    # Serialized training metadata stores references to these statistic methods.
    # They are ignored by the model and cannot execute imported MMEngine code.
    @staticmethod
    def min(*args,**kwargs):return None
    max=mean=current=min

def history_getattr(obj,name):
    if obj is not HistoryBuffer or name not in {'min','max','mean','current'}:
        raise pickle.UnpicklingError('Not allowed metadata getattr')
    return getattr(obj,name)

def metadata_encode(value,encoding='latin1',errors='strict'):
    if not isinstance(value,str) or encoding not in {'latin1','latin-1','utf8','utf-8'} or errors!='strict':
        raise pickle.UnpicklingError('Not allowed metadata encoding')
    return codecs.encode(value,encoding,errors)

class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('collections','OrderedDict'):collections.OrderedDict,
            ('mmengine.config.config','ConfigDict'):ConfigDict,
            ('mmengine.logging.history_buffer','HistoryBuffer'):HistoryBuffer,
            ('__builtin__','getattr'):history_getattr,('builtins','getattr'):history_getattr,
            ('_codecs','encode'):metadata_encode,
            ('numpy','dtype'):np.dtype,('numpy','ndarray'):np.ndarray,
            ('torch._utils','_rebuild_tensor_v2'):torch._utils._rebuild_tensor_v2}
        for namespace in ['numpy.core.multiarray','numpy._core.multiarray']:
            allowed[(namespace,'_reconstruct')]=np.core.multiarray._reconstruct
            allowed[(namespace,'scalar')]=np.core.multiarray.scalar
        for storage in ['FloatStorage','BFloat16Storage','LongStorage']:
            allowed[('torch',storage)]=getattr(torch,storage)
        if (module,name) not in allowed:raise pickle.UnpicklingError('Not allowed pickle global '+module+'.'+name)
        return allowed[(module,name)]

RESTRICTED=types.SimpleNamespace(Unpickler=RestrictedUnpickler,__name__='ground3d_restricted_pickle')

def load_official_checkpoint(path):
    # PyTorch weights_only cannot parse the official MMEngine history metadata.
    # A custom restricted unpickler accepts only the exact official data globals.
    return torch.load(path,map_location='cpu',pickle_module=RESTRICTED,weights_only=False,mmap=True)
