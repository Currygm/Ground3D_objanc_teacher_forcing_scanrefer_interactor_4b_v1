import torch
from torch import nn
from ground3d_migration.language import GroundObjAncLM

def test_frozen_visual_eval_does_not_freeze_interactor():
    lm=GroundObjAncLM.__new__(GroundObjAncLM);nn.Module.__init__(lm)
    class Language(nn.Module):
        def __init__(self):
            super().__init__();self.model=nn.Module();self.model.visual=nn.Linear(1,1)
        def get_base_model(self):return self
    lm.language=Language()
    lm.visual=nn.BatchNorm1d(256).requires_grad_(False)
    lm.interactor=nn.Linear(256,2560)
    lm.train()
    assert not lm.visual.training
    assert lm.interactor.training and all(p.requires_grad for p in lm.interactor.parameters())
    lm.eval();assert not lm.interactor.training
