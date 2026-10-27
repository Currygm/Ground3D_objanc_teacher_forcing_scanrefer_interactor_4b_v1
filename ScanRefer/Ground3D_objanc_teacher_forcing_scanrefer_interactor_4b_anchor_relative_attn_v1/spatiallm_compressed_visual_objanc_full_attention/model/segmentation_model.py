"""OBJ/ANC Qwen conditioning plus nine-layer UniVLG segmentation."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .answer_format import parse_answer
from .curriculum import normalize_phrase
from ground3d_migration.language import GroundObjAncLM as CompressedVisualObjAncLM
from .segmentation_decoder import ObjAncFullCycleDecoder

OBJECTIVE = "ground3d_objanc_teacher_forcing_scanrefer_interactor_4b_v1"
MAX_MATCH_POINTS = 50_000


def mean_sigmoid_scores(logits: torch.Tensor, positive: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    counts = positive.sum(-1)
    if bool((counts[valid] == 0).any()):
        raise ValueError("Every valid entity needs positive Answer phrase tokens")
    score = (logits.float().sigmoid()[:, :, None] * positive[:, None].float()).sum(-1)
    score = score / counts.clamp_min(1)[:, None]
    return score.masked_fill(~valid[:, None], 0).to(logits.dtype)


def dice_cost(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probability, targets = inputs.float().sigmoid(), targets.float()
    numerator = 2 * torch.einsum("qn,mn->qm", probability, targets)
    denominator = probability.sum(-1)[:, None] + targets.sum(-1)[None]
    return 1 - (numerator + 1) / (denominator + 1)


def bce_cost(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    positive = F.binary_cross_entropy_with_logits(inputs.float(), torch.ones_like(inputs).float(), reduction="none")
    negative = F.binary_cross_entropy_with_logits(inputs.float(), torch.zeros_like(inputs).float(), reduction="none")
    return (torch.einsum("qn,mn->qm", positive, targets.float()) + torch.einsum("qn,mn->qm", negative, 1-targets.float())) / inputs.shape[1]


class ObjAncSegmentationModel(nn.Module):
    objective = OBJECTIVE

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.answer_lm = CompressedVisualObjAncLM(config)
        if config['initialization'] != {'mode':'official_ground3d_3d'}:
            raise ValueError('Only official Ground3D 3D initialization is supported')
        self.answer_lm.visual.requires_grad_(True)
        self.text_adapter=nn.Linear(2560,2048)
        self.visual_adapter=nn.Linear(256,512)
        decoder = config["decoder"]
        self.decoder = ObjAncFullCycleDecoder(
            visual_dim=int(decoder["visual_dim"]), text_dim=int(decoder["text_dim"]),
            hidden_dim=int(decoder["hidden_dim"]), num_queries=int(decoder["num_queries"]),
            num_layers=int(decoder["num_layers"]), num_heads=int(decoder["num_heads"]),
            feedforward_dim=int(decoder["feedforward_dim"]), dropout=float(decoder["dropout"]),
            max_text_tokens=int(decoder["max_text_tokens"]),
            mask_threshold=float(decoder["attention_mask_threshold"]), text_temperature=float(decoder["text_temperature"]),
            anchor_config=config.get("anchor_relative_attention"),
        )
        self.loss_cfg, self.matcher_cfg, self.ranking_cfg = config["loss"], config["matcher"], config["ranking"]
        self.mask_threshold = float(config["validation"]["mask_threshold"])
        self.answer_lm_weight = float(config["training"]["answer_lm_weight"])
        self.seed = int(config["training"]["seed"])
        self._checkpoint_names = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        self._assert_boundary()

    def train(self,mode=True):
        super().train(mode)
        self.answer_lm.visual.train(mode)
        return self

    def _encode_points(self,batch):
        return self.answer_lm.encode_batch_points(batch)

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

    def _choose_raw_answers(self, batch, answers):
        """Measure original generated text against GT, including incorrect OBJ phrases."""
        target_rows, entity_indices = [], []
        for index, answer in enumerate(answers):
            parsed = parse_answer(answer)
            obj = [i for i,e in enumerate(parsed.entities) if e.role == 'OBJ']
            if not parsed.valid or len(obj) != 1:
                raise ValueError('Raw generated answer has no valid single OBJ')
            rows, indices = [0], [obj[0]]
            for i,e in enumerate(parsed.entities):
                if e.role != 'ANC': continue
                matches = [r for r,(role,phrases) in enumerate(zip(batch['grounding_roles'][index],batch['grounding_phrases'][index]))
                           if role == 'ANC' and normalize_phrase(phrases[0]) == normalize_phrase(e.phrase)]
                if len(matches)==1 and matches[0] not in rows:
                    rows.append(matches[0]); indices.append(i)
            target_rows.append(rows); entity_indices.append(indices)
        return list(answers),target_rows,entity_indices,['raw_generated']*len(answers),list(answers),0.

    def _total_loss(self, primary, auxiliary_losses, ranking, lm_loss):
        auxiliary = torch.stack(auxiliary_losses).sum()
        total = primary + float(self.loss_cfg['aux_weight'])*auxiliary + float(self.ranking_cfg['weight'])*ranking + self.answer_lm_weight*lm_loss
        return total, auxiliary

    def _connect_trainable_graph(self, total):
        return total + sum(p.reshape(-1)[0]*0 for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _positive_maps(context, entity_indices):
        text_rows = [torch.cat([q, a], dim=0) for q, a in zip(context.question_hidden, context.answer_hidden)]
        max_entities = max(len(x) for x in entity_indices)
        max_text = max(x.shape[0] for x in text_rows)
        device = text_rows[0].device
        positive = torch.zeros((len(text_rows), max_entities, max_text), device=device, dtype=torch.bool)
        valid = torch.zeros((len(text_rows), max_entities), device=device, dtype=torch.bool)
        for batch_index, indices in enumerate(entity_indices):
            spans = context.phrase_spans[batch_index].to(device)
            selected = spans[torch.tensor(indices, device=device, dtype=torch.long)]
            positive[batch_index, :len(indices), :selected.shape[1]] = selected
            valid[batch_index, :len(indices)] = True
        return text_rows, positive, valid

    def _layer_loss(self, output, targets, positive, valid):
        logits, text_valid = output["pred_logits"], output["text_valid_mask"]
        scores = mean_sigmoid_scores(logits, positive.to(logits.device), valid.to(logits.device))
        assignments, bces, dices = [], [], []
        for batch_index, target in enumerate(targets):
            length = target.shape[1]
            masks = output["pred_masks"][batch_index, :, :length]
            target = target.to(masks.device, masks.dtype)
            sample = torch.randperm(length, device=masks.device)[:MAX_MATCH_POINTS] if length > MAX_MATCH_POINTS else torch.arange(length, device=masks.device)
            cost = (-float(self.matcher_cfg["span_weight"]) * scores[batch_index, :, :target.shape[0]]
                    + float(self.matcher_cfg["mask_bce_weight"]) * bce_cost(masks[:, sample], target[:, sample])
                    + float(self.matcher_cfg["dice_weight"]) * dice_cost(masks[:, sample], target[:, sample]))
            q, t = linear_sum_assignment(cost.detach().float().cpu().numpy())
            q, t = torch.as_tensor(q, device=masks.device), torch.as_tensor(t, device=masks.device)
            assignments.append((q, t))
            selected, truth = masks[q], target[t]
            bces.append(F.binary_cross_entropy_with_logits(selected.float(), truth.float(), reduction="none").mean(-1).sum())
            prob = selected.float().sigmoid(); dices.append((1-(2*(prob*truth).sum(-1)+1)/(prob.sum(-1)+truth.sum(-1)+1)).sum())
        token_target = torch.zeros_like(logits)
        for b, (q, t) in enumerate(assignments):
            token_target[b, q] = positive.to(
                device=logits.device,
                dtype=token_target.dtype,
            )[b, t]
        active = text_valid[:, None].expand_as(logits)
        raw = F.binary_cross_entropy_with_logits(logits[active].float(), token_target[active].float(), reduction="none")
        probability = logits[active].float().sigmoid(); truth = token_target[active].float()
        pt = probability*truth + (1-probability)*(1-truth)
        alpha = float(self.loss_cfg["focal_alpha"])*truth + (1-float(self.loss_cfg["focal_alpha"]))*(1-truth)
        span = (raw * (1-pt).pow(float(self.loss_cfg["focal_gamma"])) * alpha).sum() / truth.sum().clamp_min(1)
        count = valid.sum().clamp_min(1).to(logits)
        bce = torch.stack(bces).sum()/count; dice = torch.stack(dices).sum()/count
        total = float(self.loss_cfg["span_weight"])*span + float(self.loss_cfg["mask_bce_weight"])*bce + float(self.loss_cfg["dice_weight"])*dice
        return total, span, bce, dice, assignments, scores

    def _ranking(self, scores, masks, targets, point_maps):
        terms, best_rows = [], []
        with torch.no_grad():
            for batch_index, target in enumerate(targets):
                mapping = point_maps[batch_index].to(masks.device, torch.long)
                predictions = masks[batch_index, :, :target.shape[1]][:, mapping].sigmoid() >= self.mask_threshold
                sample_best = []
                for entity in target:
                    truth = entity.to(predictions.device, torch.bool)[None]
                    if truth.shape[1] != predictions.shape[1]:
                        raise ValueError("Raw entity mask does not align with point_to_superpoint")
                    iou = (predictions & truth).sum(-1).float() / (predictions | truth).sum(-1).clamp_min(1).float()
                    sample_best.append(iou.argmax())
                best_rows.append(torch.stack(sample_best))
        for batch_index, best in enumerate(best_rows):
            for entity, best_query in enumerate(best):
                terms.append(F.cross_entropy(scores[batch_index, :, entity].float()[None] / float(self.ranking_cfg["temperature"]), best_query[None]))
        return torch.stack(terms).mean(), best_rows

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
            lm={'context':context,'loss':zero,'metrics':{'token_accuracy':zero}}

        context = lm["context"]
        text_rows, positive, valid = self._positive_maps(context, entity_indices)
        text_rows = [self.text_adapter(x.to(self.text_adapter.weight.dtype)) for x in text_rows]
        target_masks = [batch["gt_superpoint_masks"][i][rows] for i, rows in enumerate(target_rows)]
        raw_targets = [batch["gt_point_masks"][i][rows] for i, rows in enumerate(target_rows)]
        decoder = self.decoder([self.visual_adapter(x.superpoint_hidden.to(self.visual_adapter.weight.dtype)) for x in point_rows], text_rows, [x.superpoint_xyz for x in point_rows], context.token_roles,
            entities=[(sp.to(positive.device),roles) for sp,roles in zip(context.phrase_spans,context.entity_roles)],
            superpoint_counts=[torch.bincount(m.to(positive.device),minlength=len(x.superpoint_xyz)).float() for m,x in zip(batch['point_to_superpoint'],point_rows)],
            sample_ids=batch['sample_id'])
        primary, span, bce, dice, assignments, scores = self._layer_loss(decoder, target_masks, positive, valid)
        auxiliary_losses = [self._layer_loss(x, target_masks, positive, valid)[0] for x in decoder["aux_outputs"]]
        ranking, best = self._ranking(scores, decoder["pred_masks"], raw_targets, batch["point_to_superpoint"])
        total, auxiliary = self._total_loss(primary, auxiliary_losses, ranking, lm['loss'])
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
        return {"loss": total, "losses": {"total": total.detach(), "span": span.detach(), "mask_bce": bce.detach(), "dice": dice.detach(), "aux": auxiliary.detach(), "ranking": ranking.detach(), "answer_lm": lm["loss"].detach()},
                "metrics": {"teacher_ratio": total.new_tensor(ratio), "generated_training_samples": total.new_tensor(0 if answer_mode == "teacher" else len(answers)), "fallback_rate": total.new_tensor(sum(x not in {"generated", "teacher"} for x in fallback)/len(fallback)), "missing_anc_mask_count": total.new_tensor(sum(batch.get("dropped_missing_entity_count", [0]))), "answer_token_accuracy": lm["metrics"]["token_accuracy"], **{k:total.new_tensor(v) for k,v in self.answer_lm.call_counts.items()}, "visual_encode_calls": total.new_tensor(1), "interactor_calls": total.new_tensor(1), "interactor_text_tokens": total.new_tensor(0), "llm_visual_tokens": total.new_tensor(sum(len(x.compressed.hidden) for x in point_rows)), "full_superpoints": total.new_tensor(sum(len(x.superpoint_hidden) for x in point_rows))},
                "raw_logits": raw_logits, "query_scores": obj_scores, "query_entity_scores": scores,
                "selected_query": selected, "matched_query": matched, "best_queries_by_entity": best, "assignments": assignments,
                "synthetic_obj": synthetic, "chosen_answers": answers, "attempted_generated_answers": attempted, "answer_sources": fallback, "target_rows": target_rows, "entity_roles": [[batch["grounding_roles"][i][r] for r in rows] for i, rows in enumerate(target_rows)], "decoder_output": decoder}

    def optimizer_parameter_groups(self):
        t=self.config['training'];groups=[];used=set()
        specifications=[('structured_tokens',lambda n:n.startswith('answer_lm.structured.'),t['structured_token_lr']),
            ('qwen_lora',lambda n:'lora_' in n and 'visual' not in n,t['qwen_lora_lr']),
            ('interactor',lambda n:n.startswith('answer_lm.interactor.'),t['interactor_lr']),
            ('ground_visual',lambda n:n.startswith('answer_lm.visual.'),t['visual_lr']),
            ('adapters',lambda n:n.startswith(('text_adapter.','visual_adapter.')),t['adapter_lr']),
            ('decoder',lambda n:n.startswith('decoder.'),t['decoder_lr'])]
        for group,predicate,lr in specifications:
            params=[p for n,p in self.named_parameters() if p.requires_grad and predicate(n) and id(p) not in used and not n.endswith(('.anchor_sem','.anchor_geo'))]
            if not params:raise RuntimeError('Empty optimizer group '+group)
            used.update(map(id,params));groups.append({'name':group,'params':params,'lr':float(lr),'weight_decay':float(t['weight_decay'])})
        scalars=[p for n,p in self.named_parameters() if p.requires_grad and n.endswith(('.anchor_sem','.anchor_geo'))]
        if scalars:
            used.update(map(id,scalars));groups.append({'name':'anchor_scalars','params':scalars,'lr':float(t['decoder_lr']),'weight_decay':0.})
        if used!={id(p) for p in self.parameters() if p.requires_grad}:raise RuntimeError('Optimizer boundary mismatch')
        return groups

    def _assert_boundary(self):
        for name,p in self.named_parameters():
            allowed=name.startswith(('decoder.','text_adapter.','visual_adapter.','answer_lm.visual.','answer_lm.structured.','answer_lm.interactor.')) or ('lora_' in name and 'visual' not in name)
            if p.requires_grad and not allowed:raise RuntimeError('Unexpected trainable parameter '+name)

    def trainable_state_dict(self):
        return {n:v.detach().cpu() for n,v in self.state_dict().items() if n in self._checkpoint_names or n.endswith(('.anchor_sem','.anchor_geo')) or n.startswith(('answer_lm.visual.','answer_lm.interactor.'))}

    def architecture_signature(self):
        from ground3d_migration.signatures import language_signature
        return {**language_signature(self.answer_lm,self.config),'objective':OBJECTIVE,
            'decoder':dict(self.config['decoder']),'adapters':{'text':[2560,2048],'visual':[256,512]},
            'anchor_relative_attention':dict(self.config.get('anchor_relative_attention',{})),'run':self.config.get('run'),'math_version':'anchor_relative_v1',
            'loss':dict(self.config['loss']),'matcher':dict(self.config['matcher']),'ranking':dict(self.config['ranking'])}
