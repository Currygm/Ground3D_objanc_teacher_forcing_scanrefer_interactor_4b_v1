"""Evaluation-only no-Interactor adapter; original task parameters/signature stay intact.

Language replay and zero-OBJ policy match the Interactor unconstrained evaluation.
Visual encoding, point2token, adapters and Decoder remain the original no-Interactor
modules. No model state is added or renamed.
"""
from types import MethodType
from typing import Any
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from spatiallm_compressed_visual_objanc_full_attention.model.llm import FullAttentionContext, ROLE_QUESTION, ROLE_OBJ, ROLE_OTHER
from spatiallm_compressed_visual_objanc_full_attention.model.answer_format import parse_answer
from reason3d_runtime import ObjAncSegmentationModel as TrainingModel

def _text_forward(self,embeds,valid,positions,*,past=None,use_cache=False):
        if not hasattr(self,'call_counts'):self.call_counts={}
        key='generation_prefill' if use_cache and past is None else ('generation_incremental' if use_cache else 'causal_forward')
        self.call_counts[key]=self.call_counts.get(key,0)+1
        return self.language.get_base_model().model.language_model(inputs_embeds=embeds,attention_mask=valid,
            position_ids=positions[None].expand(3,-1,-1),past_key_values=past,use_cache=use_cache,return_dict=True)

def _causal(self,questions,answers,visuals,*,synthetic_obj=None,compute_loss=True):
        if not(len(questions)==len(answers)==len(visuals)):raise ValueError('Causal batch lengths differ')
        synthetic_obj=[False]*len(answers) if synthetic_obj is None else list(synthetic_obj)
        if len(synthetic_obj)!=len(answers) or (compute_loss and any(synthetic_obj)):raise ValueError('Synthetic OBJ is evaluation-only')
        rows=[];labels=[];maps=[];parsed_rows=[]
        for question,answer,visual,synthetic in zip(questions,answers,visuals,synthetic_obj):
            parsed=parse_answer(answer)
            if not parsed.valid and not synthetic:raise ValueError('Invalid OBJ/ANC answer '+str(parsed.errors))
            prompt_ids,qidx=self._prompt_tokens(question)
            encoded=self.tokenizer(answer,add_special_tokens=False,return_offsets_mapping=True,return_tensors='pt')
            aids=encoded['input_ids'][0];all_a=torch.cat([aids,aids.new_tensor([self.eos_token_id])])
            ids=torch.cat([prompt_ids,all_a]);merged,pos,marker,shift=self._merge(ids,visual)
            label=torch.full((len(merged),),-100,device=merged.device,dtype=torch.long)
            start=len(prompt_ids)+shift;label[start:]=all_a.to(merged.device)
            if synthetic:
                roles=torch.full((len(aids),),ROLE_OTHER,dtype=torch.long)
                spans=torch.zeros((0,len(aids)),dtype=torch.bool)
            else:
                roles,spans=self._answer_roles(answer,encoded['offset_mapping'][0],parsed)
            mapped_q=qidx+(qidx>marker).long()*shift
            maps.append((mapped_q.to(merged.device),torch.arange(len(aids),device=merged.device)+start,roles,spans))
            rows.append(merged);labels.append(label);parsed_rows.append(parsed)
        embeds=pad_sequence(rows,batch_first=True)
        valid=pad_sequence([torch.ones(len(x),device=embeds.device,dtype=torch.long) for x in rows],batch_first=True)
        positions=pad_sequence([torch.arange(len(x),device=embeds.device) for x in rows],batch_first=True)
        output=self._text_forward(embeds,valid,positions)
        if compute_loss:
            target=pad_sequence(labels,batch_first=True,padding_value=-100)[:,1:];selected=target.ne(-100)
            logits=self.structured.logits(output.last_hidden_state[:,:-1][selected],self.base.lm_head)
            loss=F.cross_entropy(logits.float(),target[selected]);accuracy=logits.argmax(-1).eq(target[selected]).float().mean()
            from reason3d_obj_loss import obj_prediction_mask,obj_content_loss
            obj_mask=obj_prediction_mask(maps,parsed_rows,target)
            obj_ce,obj_metrics=obj_content_loss(logits,target[selected],obj_mask[selected])
        else:
            loss=output.last_hidden_state.new_zeros(());accuracy=loss.detach()
            obj_ce=loss;obj_metrics={'obj_token_accuracy':accuracy,'obj_token_count':accuracy}
        qh=[];ah=[];rs=[];ss=[];ers=[]
        for b,(qi,ai,role,span) in enumerate(maps):
            q=output.last_hidden_state[b,qi];a=output.last_hidden_state[b,ai]
            if synthetic_obj[b]:
                a=torch.cat([a,a.new_zeros(1,a.shape[-1])],dim=0)
                role=torch.cat([role,role.new_tensor([ROLE_OBJ])])
                span=torch.zeros((1,len(q)+len(a)),dtype=torch.bool);span[0,-1]=True
                entity_roles=('OBJ',)
            else:
                span=torch.cat([torch.zeros(len(span),len(q),dtype=torch.bool),span],dim=1)
                entity_roles=tuple(e.role for e in parsed_rows[b].entities)
            qh.append(q);ah.append(a)
            rs.append(torch.cat([torch.full((len(q),),ROLE_QUESTION,dtype=torch.long),role]))
            ss.append(span);ers.append(entity_roles)
        return {'loss':loss,'losses':{'answer_obj_ce':obj_ce},'metrics':{'token_accuracy':accuracy,**obj_metrics},'context':FullAttentionContext(qh,ah,rs,ss,ers,parsed_rows)}

def encode_full_context(self,point_clouds,questions,answers,*,visuals=None,synthetic_obj=None):
        if visuals is None:raise ValueError('Missing Ground3D visual features')
        return self._causal(questions,answers,visuals,synthetic_obj=synthetic_obj,compute_loss=False)['context']

def install_language(lm):
    """Bind evaluation methods without modifying parameter names or architecture."""
    for name in ("_text_forward","_causal","encode_full_context"):
        setattr(lm,name,MethodType(globals()[name],lm))

class ObjAncSegmentationModel(TrainingModel):
    """Native no-Interactor model with raw generation and zero-OBJ evaluation."""
    def _choose_answers(self, batch, point_rows, answer_mode):
            if answer_mode=='teacher':
                return (list(batch['teacher_answer']),[list(range(len(x))) for x in batch['grounding_object_ids']],
                    [list(map(int,x)) for x in batch['answer_entity_indices']],['teacher']*len(point_rows),
                    [None]*len(point_rows),1.)
            if answer_mode!='generated':raise ValueError('Unsupported answer mode '+answer_mode)
            answers=[];rows=[];indices=[];reasons=[]
            for i,(question,features) in enumerate(zip(batch['question'],point_rows)):
                with torch.no_grad():generated=self.answer_lm.generate_one(question,visual=features.compressed)
                answer=generated['answer'];parsed=parse_answer(answer)
                failed=generated['format_failure'] or not parsed.valid or parsed.obj is None
                if failed:
                    reason='budget_failure' if generated['budget_forced'] else 'format_failure'
                    rows.append([0]);indices.append([0]);reasons.append('zero_obj:'+reason)
                else:
                    # GT phrases align optional ANC metrics only, never gate OBJ inference.
                    single={k:[v[i]] for k,v in batch.items()}
                    _,r,e,_,_,_=self._choose_raw_answers(single,[answer])
                    rows.append(r[0]);indices.append(e[0]);reasons.append('generated')
                answers.append(answer)
            return answers,rows,indices,reasons,list(answers),0.

    def forward(
            self,
            batch: dict[str, list[Any]],
            *,
            epoch: int,
            optimizer_step: int,
            steps_per_epoch: int,
            answer_mode: str = "teacher",
            generated_answers: list[str] | None = None,
        ):
            if self.training and (
                answer_mode != "teacher"
                or bool(self.config["training"].get("allow_generated_answers", False))
            ):
                raise RuntimeError(
                    "Training is strictly teacher-forced; generated answers are disabled"
                )
            self.answer_lm.call_counts={}
            point_rows = self._encode_points(batch)
            visuals = [row.compressed for row in point_rows]
            if answer_mode == 'raw_generated':
                if generated_answers is None:raise ValueError('Raw evaluation requires cached generated answers')
                answers, target_rows, entity_indices, fallback, attempted, ratio = self._choose_raw_answers(batch,generated_answers)
            else:
                answers, target_rows, entity_indices, fallback, attempted, ratio = self._choose_answers(batch, point_rows, answer_mode)
            synthetic=[reason.startswith('zero_obj:') for reason in fallback]
            if answer_mode=='teacher':
                lm=self.answer_lm(batch['point_cloud'],batch['question'],answers,visuals=visuals)
            else:
                context=self.answer_lm.encode_full_context(batch['point_cloud'],batch['question'],answers,visuals=visuals,synthetic_obj=synthetic)
                zero=point_rows[0].superpoint_hidden.new_zeros(())
                lm={'context':context,'loss':zero,'losses':{'answer_obj_ce':zero},'metrics':{'token_accuracy':zero,'obj_token_accuracy':zero,'obj_token_count':zero}}

            context = lm["context"]
            text_rows, positive, valid = self._positive_maps(context, entity_indices)
            text_rows = [self.text_adapter(x.to(self.text_adapter.weight.dtype)) for x in text_rows]
            target_masks = [batch["gt_superpoint_masks"][i][rows] for i, rows in enumerate(target_rows)]
            raw_targets = [batch["gt_point_masks"][i][rows] for i, rows in enumerate(target_rows)]
            decoder = self.decoder([self.visual_adapter(x.superpoint_hidden.to(self.visual_adapter.weight.dtype)) for x in point_rows], text_rows, [x.superpoint_xyz for x in point_rows], context.token_roles)
            primary, span, bce, dice, assignments, scores = self._layer_loss(decoder, target_masks, positive, valid)
            auxiliary_losses = [self._layer_loss(x, target_masks, positive, valid)[0] for x in decoder["aux_outputs"]]
            ranking, best = self._ranking(scores, decoder["pred_masks"], raw_targets, batch["point_to_superpoint"])
            total, auxiliary = self._total_loss(primary, auxiliary_losses, ranking, lm['loss'])
            obj_ce=lm['losses']['answer_obj_ce']
            total=total+self.obj_ce_weight*obj_ce
            if self.training:
                # Conditional text spans and final updates may leave parameters unused on a rank.
                # Zero connections preserve objective values and make DDP reduction complete.
                total = self._connect_trainable_graph(total)
            obj_scores = scores[:, :, 0]; selected = obj_scores.argmax(1)
            matched = []
            for source, target in assignments:
                position = torch.where(target.eq(0))[0]
                matched.append(source[position[0]])
            matched = torch.stack(matched)
            selected_logits = [decoder["pred_masks"][i, selected[i], :target_masks[i].shape[1]] for i in range(len(target_masks))]
            raw_logits = [x[m.to(x.device)] for x, m in zip(selected_logits, batch["point_to_superpoint"])]
            return {"loss": total, "losses": {"total": total.detach(), "span": span.detach(), "mask_bce": bce.detach(), "dice": dice.detach(), "aux": auxiliary.detach(), "ranking": ranking.detach(), "answer_lm": lm["loss"].detach(), "answer_obj_ce":obj_ce.detach(), "answer_contribution":self.answer_lm_weight*lm["loss"].detach(), "obj_ce_contribution":self.obj_ce_weight*obj_ce.detach()},
                    "metrics": {"teacher_ratio": total.new_tensor(ratio), "generated_training_samples": total.new_tensor(0 if answer_mode == "teacher" else len(answers)), "fallback_rate": total.new_tensor(sum(x not in {"generated", "teacher"} for x in fallback)/len(fallback)), "missing_anc_mask_count": total.new_tensor(sum(batch.get("dropped_missing_entity_count", [0]))), "answer_token_accuracy": lm["metrics"]["token_accuracy"], "answer_obj_token_accuracy":lm["metrics"]["obj_token_accuracy"], "answer_obj_token_count":lm["metrics"]["obj_token_count"], **{k:total.new_tensor(v) for k,v in self.answer_lm.call_counts.items()}, "visual_encode_calls": total.new_tensor(1), "interactor_calls": total.new_tensor(0), "interactor_text_tokens": total.new_tensor(0), "llm_visual_tokens": total.new_tensor(sum(len(x.compressed.hidden) for x in point_rows)), "full_superpoints": total.new_tensor(sum(len(x.superpoint_hidden) for x in point_rows))},
                    "raw_logits": raw_logits, "query_scores": obj_scores, "query_entity_scores": scores,
                    "selected_query": selected, "matched_query": matched, "best_queries_by_entity": best, "assignments": assignments,
                    "synthetic_obj": synthetic, "chosen_answers": answers, "attempted_generated_answers": attempted, "answer_sources": fallback, "target_rows": target_rows, "entity_roles": [[batch["grounding_roles"][i][r] for r in rows] for i, rows in enumerate(target_rows)], "decoder_output": decoder}
