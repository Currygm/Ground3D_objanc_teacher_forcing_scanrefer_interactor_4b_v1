from types import SimpleNamespace
import pytest
import torch
from torch import nn
from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_model import ObjAncSegmentationModel

@pytest.mark.parametrize('mode',['teacher','generated','raw_generated'])
def test_segmentation_replays_only_chosen_answer_once(mode):
    events=[];teacher='<obj>chair</obj>';generated='<obj>red chair</obj>'
    class LM(nn.Module):
        def __init__(self):super().__init__();self.visual=nn.Identity()
        def forward(self,clouds,questions,answers,*,visuals):
            events.append(('causal',list(answers)));return {'context':SimpleNamespace(), 'loss':torch.tensor(0.),'metrics':{}}
        def encode_full_context(self,clouds,questions,answers,*,visuals):events.append(('replay',list(answers)));return SimpleNamespace()
    model=ObjAncSegmentationModel.__new__(ObjAncSegmentationModel);nn.Module.__init__(model)
    model.answer_lm=LM();model.eval()
    model._encode_points=lambda batch:[SimpleNamespace(compressed=None)]
    def choose(batch,rows,kind):
        events.append(('choose',kind));answer=teacher if kind=='teacher' else generated
        return [answer],[[0]],[[0]],['teacher' if kind=='teacher' else 'generated'],[None],float(kind=='teacher')
    model._choose_answers=choose
    model._choose_raw_answers=lambda batch,answers:choose(batch,None,'raw_generated')
    class StopBeforeDecoder(Exception):pass
    def stop(context,indices):raise StopBeforeDecoder()
    model._positive_maps=stop
    batch={'point_cloud':[torch.zeros(2,6)],'question':['chair?'],'teacher_answer':[teacher]}
    with pytest.raises(StopBeforeDecoder):model(batch,epoch=0,optimizer_step=0,steps_per_epoch=1,answer_mode=mode,generated_answers=[generated])
    assert events==[('choose',mode),('causal',[teacher if mode=='teacher' else generated])]
