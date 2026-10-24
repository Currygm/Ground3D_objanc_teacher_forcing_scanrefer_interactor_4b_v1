"""Three ScanRefer-only epochs of causal Answer SFT before joint training."""
from __future__ import annotations
import torch
from torch import nn
from ground3d_migration.language import GroundObjAncLM as CompressedVisualObjAncLM

ANSWER_OBJECTIVE='ground3d_objanc_teacher_forcing_scanrefer_4b_answer_v1'

def answer_signature(language,config):
    from ground3d_migration.signatures import language_signature
    return {**language_signature(language,config),'objective':ANSWER_OBJECTIVE}

def is_answer_parameter(name):
    return name.startswith('answer_lm.structured.') or (
        name.startswith('answer_lm.') and 'lora_' in name and 'sonata_lora_' not in name
        and 'point_backbone' not in name)

class AnswerPretrainingStage(nn.Module):
    objective=ANSWER_OBJECTIVE

    def __init__(self,config):
        super().__init__()
        self.config=config; self.answer_lm=CompressedVisualObjAncLM(config)
        for name,parameter in self.named_parameters():
            if parameter.requires_grad and not is_answer_parameter(name):
                raise RuntimeError(f'Unexpected Answer-stage trainable parameter: {name}')

    def forward(self,batch,*,epoch=0,optimizer_step=0,steps_per_epoch=1,answer_mode='teacher'):
        if answer_mode!='teacher':raise ValueError('Answer pretraining is teacher-forced SFT only')
        rows=self.answer_lm.encode_batch_points(batch)
        output=self.answer_lm(batch['point_cloud'],batch['question'],batch['teacher_answer'],visuals=[x.compressed for x in rows])
        loss=output['loss']
        if self.training:
            loss=loss+sum(p.reshape(-1)[0]*0 for p in self.parameters() if p.requires_grad)
        return {'loss':loss,'losses':{'total':loss.detach(),'answer_lm':output['loss'].detach()},
                'metrics':{'answer_token_accuracy':output['metrics']['token_accuracy'],'llm_visual_tokens':torch.tensor(sum(len(x.compressed.hidden) for x in rows)/len(rows)), 'full_superpoints':torch.tensor(sum(len(x.superpoint_hidden) for x in rows)/len(rows))}}

    def architecture_signature(self):return answer_signature(self.answer_lm,self.config)

    def trainable_state_dict(self):
        return {n:p.detach().cpu() for n,p in self.named_parameters() if p.requires_grad}

    def optimizer_parameter_groups(self):
        training=self.config['training']; groups=[]
        for group,structured,key in [('qwen_lora',False,'qwen_lora_lr'),('structured_tokens',True,'structured_token_lr')]:
            parameters=[p for n,p in self.named_parameters() if p.requires_grad
                        and n.startswith('answer_lm.structured.')==structured]
            if not parameters:raise RuntimeError(f'No trainable parameters for {group}')
            groups.append({'name':group,'params':parameters,'lr':float(training[key]),
                           'weight_decay':float(training['weight_decay'])})
        return groups
