"""Immutable per-sample first-pass evidence with split and checksum contracts."""
import json
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from runtime_helpers import atomic_save,sha256
from .objects import select_object_evidence

CACHE_PROTOCOL="fixed_objects_textscore_v3"
FULL_CACHE_PROTOCOL="fixed_objects_full_scene_v4"

def _cpu(value):
    return value.detach().cpu().clone()

@torch.no_grad()
def make_record(proposal,batch,row,language,*,index,include_full_scene=False):
    objects=select_object_evidence(proposal)
    question=batch["question"][row]
    ids=language.tokenizer.encode(question,add_special_tokens=False)
    if not ids:raise ValueError("Empty question")
    device=next(language.language.parameters()).device
    tokens=torch.tensor(ids,device=device,dtype=torch.long)
    question_mean=language.structured.embed(tokens,language.language.get_base_model().get_input_embeddings()).mean(0)
    record={"protocol":FULL_CACHE_PROTOCOL if include_full_scene else CACHE_PROTOCOL,"index":int(index),"sample_id":str(batch["sample_id"][row]),
        "question":question,"answer":proposal.answer,"failed":bool(proposal.failed),
        "scene":_cpu(proposal.visual.hidden),"question_mean":_cpu(question_mean),
        "evidence":{"features":_cpu(objects.features),"roles":objects.roles,"phrases":objects.phrases,
            "source_query_ids":_cpu(objects.source_query_ids),"valid_mask":_cpu(objects.valid_mask)},
        "scoring_keys":_cpu(proposal.scoring_keys),"text_temperature":proposal.text_temperature,
        "masks":_cpu(proposal.masks).float(),"first_scores":_cpu(proposal.entity_scores[:,proposal.obj_index]).float(),
        "mapping":_cpu(proposal.point_to_superpoint).long(),
        # Labels live separately from inference evidence and are never used by predict().
        "truth":_cpu(batch["gt_point_masks"][row][0]).bool()}
    if include_full_scene:
        record.update(superpoint_features=_cpu(proposal.features),superpoint_xyz=_cpu(proposal.xyz),
            supervision={"teacher_answer":str(batch["teacher_answer"][row]),
                "answer_entity_indices":list(map(int,batch["answer_entity_indices"][row])),
                "gt_point_masks":_cpu(batch["gt_point_masks"][row]).bool(),
                "gt_superpoint_masks":_cpu(batch["gt_superpoint_masks"][row]).float()})
    return record

def validate_record(value,index=None,*,require_full_scene=False):
    expected=FULL_CACHE_PROTOCOL if require_full_scene else CACHE_PROTOCOL
    if value.get("protocol")!=expected or (index is not None and value.get("index")!=index):
        raise ValueError("Wrong cache protocol/index")
    masks=value["masks"];scores=value["first_scores"];mapping=value["mapping"];truth=value["truth"]
    e=value["evidence"];k=len(e["roles"])
    if (masks.ndim!=2 or masks.shape[0]!=50 or masks.shape[1]<1 or scores.shape!=(50,)
        or value["scoring_keys"].ndim!=2 or value["scoring_keys"].shape[0]!=50
        or value["text_temperature"]<=0
        or mapping.ndim!=1 or truth.shape!=mapping.shape or not len(mapping)
        or mapping.min()<0 or mapping.max()>=masks.shape[1] or not value["sample_id"]
        or e["features"].ndim!=2 or len(e["features"])!=k or len(e["phrases"])!=k
        or e["source_query_ids"].shape!=(k,) or e["valid_mask"].shape!=(k,) or e["valid_mask"].dtype!=torch.bool
        or e["source_query_ids"].dtype!=torch.long or mapping.dtype!=torch.long or truth.dtype!=torch.bool
        or (k and (e["source_query_ids"].min()<0 or e["source_query_ids"].max()>=50))):
        raise ValueError("Invalid cache tensor/entity contract")
    for tensor in [masks,scores,value["scoring_keys"],value["scene"],value["question_mean"],e["features"]]:
        if not torch.isfinite(tensor).all():raise ValueError("Nonfinite cached evidence")
    if value["scene"].ndim!=2 or value["question_mean"].shape!=(value["scene"].shape[-1],):
        raise ValueError("Invalid cached language dimensions")
    if require_full_scene:
        from spatiallm_compressed_visual_objanc_full_attention.model.answer_format import parse_answer
        f=value.get("superpoint_features");xyz=value.get("superpoint_xyz");sup=value.get("supervision",{})
        if (not isinstance(f,torch.Tensor) or f.shape!=(masks.shape[1],e["features"].shape[1])
            or not isinstance(xyz,torch.Tensor) or xyz.shape!=(masks.shape[1],3)
            or not torch.isfinite(f).all() or not torch.isfinite(xyz).all()):
            raise ValueError("Stage B requires aligned full-scene superpoint features/XYZ")
        raw=sup.get("gt_point_masks");sp=sup.get("gt_superpoint_masks");indices=sup.get("answer_entity_indices")
        parsed=parse_answer(sup.get("teacher_answer",""))
        if (not parsed.valid or not isinstance(indices,list) or not indices or len(set(indices))!=len(indices)
            or any(not isinstance(i,int) or i<0 or i>=len(parsed.entities) for i in indices)
            or parsed.entities[indices[0]].role!="OBJ"
            or not isinstance(raw,torch.Tensor) or raw.shape!=(len(indices),len(mapping))
            or not isinstance(sp,torch.Tensor) or sp.shape!=(len(indices),masks.shape[1])
            or not torch.isfinite(sp).all() or not torch.equal(raw[0].bool(),truth)):
            raise ValueError("Stage B supervision/text/entity/mask alignment differs")
    return value

def _atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    os.replace(tmp,path)

class CacheWriter:
    def __init__(self,root,split,identity,count,*,resume=False,full_scene=False):
        self.root=Path(root);self.split=split;self.identity=identity;self.count=int(count)
        self.full_scene=bool(full_scene);self.protocol=FULL_CACHE_PROTOCOL if full_scene else CACHE_PROTOCOL
        if split not in {"train","val"} or self.count<1:raise ValueError("Invalid cache split/count")
        self.root.mkdir(parents=True,exist_ok=True)
        metadata={"protocol":self.protocol,"split":split,"identity":identity,"count":self.count}
        draft=self.root/"draft.json"
        if draft.exists():
            if not resume:raise FileExistsError("Cache exists; use --resume to validate/reuse")
            if json.loads(draft.read_text())!=metadata:raise ValueError("Cache resume source changed")
        else:_atomic_json(draft,metadata)
        (self.root/"records").mkdir(exist_ok=True)

    def write(self,index,value):
        if not 0<=index<self.count:raise ValueError("Cache index out of range")
        validate_record(value,index,require_full_scene=self.full_scene)
        path=self.root/"records"/f"{index:08d}.pt"
        if path.exists():raise FileExistsError("Refusing to overwrite cache record")
        atomic_save(value,path)
        _atomic_json(path.with_suffix(".checksum.json"),{"sha256":sha256(path),"index":index})
        return path

    def validate_existing(self,index):
        path=self.root/"records"/f"{index:08d}.pt"
        proof=path.with_suffix(".checksum.json")
        if not proof.is_file() or json.loads(proof.read_text())!={"sha256":sha256(path),"index":index}:
            raise ValueError("Uncommitted or modified cache record checksum")
        return validate_record(torch.load(path,map_location="cpu",weights_only=False),index,require_full_scene=self.full_scene)

    def finalize(self):
        entries=[];ids=set()
        for index in range(self.count):
            path=self.root/"records"/f"{index:08d}.pt"
            if not path.is_file():raise ValueError("Incomplete first-stage cache")
            value=self.validate_existing(index)
            if value["sample_id"] in ids:raise ValueError("Duplicate cache sample ID")
            ids.add(value["sample_id"])
            entries.append({"index":index,"sample_id":value["sample_id"],"file":str(path.relative_to(self.root)),
                "sha256":sha256(path),"bytes":path.stat().st_size})
        result={"protocol":self.protocol,"split":self.split,"identity":self.identity,
                "count":self.count,"entries":entries}
        _atomic_json(self.root/"manifest.json",result)
        return result

class CachedDataset(Dataset):
    def __init__(self,root,identity,*,expected_count,split,full_scene=False):
        self.root=Path(root);self.full_scene=bool(full_scene)
        protocol=FULL_CACHE_PROTOCOL if full_scene else CACHE_PROTOCOL
        manifest=self.root/"manifest.json"
        if not manifest.is_file():raise ValueError("Complete cache manifest is required")
        self.manifest=json.loads(manifest.read_text())
        m=self.manifest
        if (m.get("protocol")!=protocol or m.get("identity")!=identity or m.get("split")!=split
            or m.get("count")!=expected_count or len(m.get("entries",[]))!=expected_count):
            raise ValueError("First-stage cache source/split/count changed")
        if [x["index"] for x in m["entries"]]!=list(range(expected_count)):
            raise ValueError("Non-contiguous cached indices")
        if len({x["sample_id"] for x in m["entries"]})!=expected_count:
            raise ValueError("Duplicate cache sample ID")
        self.manifest_sha256=sha256(manifest)
    def __len__(self):return self.manifest["count"]
    def __getitem__(self,index):
        entry=self.manifest["entries"][index]
        path=(self.root/entry["file"]).resolve(strict=True)
        if not path.is_relative_to(self.root.resolve()) or sha256(path)!=entry["sha256"]:
            raise ValueError("Cache record checksum/path changed")
        value=validate_record(torch.load(path,map_location="cpu",weights_only=False),index,require_full_scene=self.full_scene)
        if value["sample_id"]!=entry["sample_id"]:raise ValueError("Cache sample identity changed")
        return value
