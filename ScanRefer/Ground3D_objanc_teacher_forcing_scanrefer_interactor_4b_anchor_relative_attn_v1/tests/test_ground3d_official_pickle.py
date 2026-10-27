import pickle,sys,types
import pytest
import torch
from ground3d_migration.checkpoint import load_official_checkpoint

def test_official_metadata_classes_do_not_require_mmengine(tmp_path,monkeypatch):
    module=types.ModuleType('mmengine.config.config')
    ConfigDict=type('ConfigDict',(dict,),{'__module__':module.__name__})
    module.ConfigDict=ConfigDict
    for name in ['mmengine','mmengine.config']:
        monkeypatch.setitem(sys.modules,name,types.ModuleType(name))
    monkeypatch.setitem(sys.modules,module.__name__,module)
    path=tmp_path/'official.pt'
    torch.save({'meta':ConfigDict(epoch=1),'state_dict':{'weight':torch.ones(2)}},path)
    monkeypatch.delitem(sys.modules,module.__name__)
    assert torch.equal(load_official_checkpoint(path)['state_dict']['weight'],torch.ones(2))

def test_reader_rejects_unlisted_pickle_globals(tmp_path):
    class Bad:
        def __reduce__(self):return (eval,('123',))
    path=tmp_path/'bad.pt';torch.save({'state_dict':{},'meta':Bad()},path)
    with pytest.raises(pickle.UnpicklingError,match='Not allowed'):load_official_checkpoint(path)
