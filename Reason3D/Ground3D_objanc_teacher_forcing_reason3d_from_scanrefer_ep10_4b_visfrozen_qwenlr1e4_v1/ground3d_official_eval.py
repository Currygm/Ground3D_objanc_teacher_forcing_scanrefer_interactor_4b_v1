"""Official Ground3D-LMM inference with matched functional validation scoring.
No GT labels, answers or target phrases are passed into official generation.
"""
from pathlib import Path
from collections import OrderedDict
import hashlib
import json
import re
import numpy as np

PROJECT=Path(__file__).resolve().parent
OFFICIAL_SHA256="cc7c379567b609d5e9dfc9001c8d6ec2ad1a252d99d3d38b6d1f27cd4f92bc69"
ANNOTATION_SHA256="5f45ef0391a527a76486ec0aabe73d4567134907cec9cfffac89b74456abcb85"

def normalize_state(source):
    """Preserve version metadata used by the native SpConv checkpoint loader."""
    import copy
    from collections import OrderedDict
    from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
    result=OrderedDict(source)
    if hasattr(source,"_metadata"):
        result._metadata=copy.deepcopy(source._metadata)
    consume_prefix_in_state_dict_if_present(result,"module.")
    return result

def validate_sparse_metadata(model,state):
    from spconv.pytorch.conv import SparseConvolution
    metadata=getattr(state,"_metadata",{})
    count=0
    for name,module in model.named_modules():
        if isinstance(module,SparseConvolution):
            if metadata.get(name,{}).get("version")!=2:
                raise ValueError("SpConv v2 layout metadata missing or incompatible: "+name)
            count+=1
    return count

def validate_state(destination, source):
    unexpected=sorted(set(source)-set(destination))
    if unexpected: raise ValueError("Unexpected checkpoint keys: "+str(unexpected[:20]))
    missing=sorted(set(destination)-set(source))
    # Released task-only checkpoint omits frozen Qwen base tensors only.
    # Embeddings, lm_head, LoRA and all 3D/decoder tensors MUST be present.
    def frozen_base(k):
        return (k.startswith("decoder.reason_model.")
            and "lora_" not in k and ".embed_tokens." not in k and ".lm_head." not in k)
    task_missing=[k for k in missing if not frozen_base(k)]
    if task_missing: raise ValueError("Missing task checkpoint keys: "+str(task_missing[:20]))
    for k,v in source.items():
        if tuple(v.shape)!=tuple(destination[k].shape):
            raise ValueError("Checkpoint shape mismatch: "+k)
    return dict(loaded_tensors=len(source),allowed_base_missing=missing,unexpected=[])

def score_prediction(prediction, truth, target_phrase):
    truth=np.asarray(truth,dtype=bool)
    answer=prediction["answer"]
    phrases=re.findall(r"<p>([^<>]+)</p>",answer)
    reason=prediction.get("reason")
    if not phrases: reason=reason or "missing_grounded_phrase"
    if prediction.get("mask") is None: reason=reason or "missing_mask"
    result=dict(valid=reason is None,generated_answer=answer,format_valid=bool(phrases),
        failure_reason=reason,selected_query=0 if reason is None else None,
        query_count=prediction.get("query_count",0),budget_forced=False,
        iou=0.,acc25=0.,acc50=0.,intersection=0,union=int(truth.sum()),
        obj_phrase_correct=False)
    if reason is None:
        mask=np.asarray(prediction["mask"],dtype=bool)
        if mask.shape!=truth.shape: raise ValueError("Prediction/GT point order differs")
        intersection=int((mask&truth).sum()); union=int((mask|truth).sum())
        iou=intersection/max(union,1)
        norm=lambda s:" ".join(s.lower().split())
        result.update(iou=iou,intersection=intersection,union=union,
            acc25=float(iou>=.25),acc50=float(iou>=.5),
            obj_phrase_correct=norm(phrases[0])==norm(target_phrase))
    return result

class OfficialDataset:
    def __init__(self,rows,data_root):
        self.rows=rows; self.data_root=Path(data_root); self.cache=OrderedDict()
    def inference(self,index):
        import torch
        from ground3d_functional import frame_path,array_path
        row=self.rows[index]; prefix=frame_path(row,self.data_root)
        if str(prefix) not in self.cache:
            vert=np.load(array_path(prefix,"vert"),allow_pickle=False)
            sp=np.load(array_path(prefix,"sp_label"),allow_pickle=False).reshape(-1)
            if vert.ndim!=2 or vert.shape[1]!=6 or len(vert)!=len(sp) or not np.isfinite(vert).all():
                raise ValueError("Malformed inference geometry")
            if not np.issubdtype(sp.dtype,np.integer): raise ValueError("Noninteger superpoints")
            points=torch.from_numpy(vert.copy()).float()
            if bool(((points[:,3:]<0)|(points[:,3:]>255)).any()): raise ValueError("Expected byte RGB")
            points[:,3:]=(points[:,3:]-127.5)/127.5
            # Same contiguous-ID normalization as official annotation loader.
            _,inverse=np.unique(sp,return_inverse=True)
            self.cache[str(prefix)]=(points,torch.from_numpy(inverse).long())
            while len(self.cache)>2:self.cache.popitem(last=False)
        points,sp=self.cache[str(prefix)]
        return dict(points=points,superpoints=sp,question=row["question"],
                    scene_name=row["scene_id"]+"_frame_"+str(row["frame"]),
                    sample_id=row["sample_id"])
    def truth(self,index):
        from ground3d_functional import frame_path,array_path
        row=self.rows[index]
        expected="object_label" if row["level"]=="object_qa" else "part_label"
        if row["target_label_field"]!=expected:raise ValueError("GT label field differs")
        labels=np.load(array_path(frame_path(row,self.data_root),expected),allow_pickle=False).reshape(-1)
        truth=np.isin(labels,row["target_ids"])
        if not truth.any():raise ValueError("Empty GT")
        return truth

def read_annotations(path, *, max_samples=None):
    path=Path(path); raw=path.read_bytes()
    manifest=json.loads(path.with_name("manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(raw).hexdigest()!=ANNOTATION_SHA256 or manifest["annotation_sha256"]!=ANNOTATION_SHA256:
        raise ValueError("Expected exactly the previously evaluated annotation manifest")
    rows=[json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
    if len(rows)!=8329 or manifest["samples"]!=8329 or len({r["sample_id"] for r in rows})!=8329:
        raise ValueError("Expected 8329 unique functional validation samples")
    counts={}
    for r in rows:
        key=r["source"]+"/"+r["level"]; counts[key]=counts.get(key,0)+1
    if counts!={"scannet/object_qa":3100,"scannet/part_qa":2606,
                "scannetpp/object_qa":1609,"scannetpp/part_qa":1014}:
        raise ValueError("Validation subgroup counts differ")
    if max_samples is not None:
        if max_samples<1:raise ValueError("--max-samples must be positive")
        rows=rows[:max_samples]
    return rows,manifest

def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def build_official(repo,checkpoint,qwen,output, *, device="cpu",verify_hash=True):
    import sys
    import torch
    repo=Path(repo).resolve(); qwen=Path(qwen).resolve()
    if verify_hash and file_sha256(checkpoint)!=OFFICIAL_SHA256:
        raise ValueError("Official 3D checkpoint SHA256 differs")
    sys.path.insert(0,str(repo))
    from mmdet3d.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    import uniseg3d
    from mmengine.config import Config
    from mmdet3d.registry import MODELS
    cfg=Config.fromfile(str(repo/"configs/ground3dlmm_eval_ground3d_scannet_3d.py"))
    cfg.model.decoder.qwen_model_path=str(qwen)
    cfg.model.decoder.save_pred_qa_dir=str(output)
    cfg.model.decoder.image_dir=None
    if (cfg.model.decoder.lora_r,cfg.model.decoder.lora_alpha)!=(32,64):
        raise ValueError("Official 3D LoRA config differs")
    # The official decoder accepts this path but never reads it in sphere_cls mode.
    cfg.model.decoder.vocabulary_cls_embedding_path=str(repo/"data/scannet/scannet_cls_embedding.pth")
    model=MODELS.build(cfg.model).eval()
    ckpt=torch.load(checkpoint,map_location="cpu",mmap=True,weights_only=False)
    state=ckpt.get("state_dict",ckpt)
    state=normalize_state(state)
    report=validate_state(model.state_dict(),state)
    report["spconv_v2_metadata_modules"]=validate_sparse_metadata(model,state)
    print("Official task keys, shapes and SpConv metadata validated",flush=True)
    for k,v in state.items():
        if torch.is_floating_point(v) and not torch.isfinite(v).all():
            raise ValueError("Nonfinite official weight: "+k)
    loaded=model.load_state_dict(state,strict=False)
    if sorted(loaded.missing_keys)!=report["allowed_base_missing"] or loaded.unexpected_keys:
        raise ValueError("Actual state loading differs from verified contract")
    from spconv.pytorch.conv import SparseConvolution
    for name,module in model.named_modules():
        if isinstance(module,SparseConvolution) and not torch.equal(module.weight.detach(),state[name+".weight"].to(module.weight.dtype)):
            raise ValueError("SpConv values changed during checkpoint loading: "+name)
    report["spconv_weights_value_equal"]=True
    # Frozen base was loaded from local Qwen by official from_pretrained.
    report.update(checkpoint=str(checkpoint),sha256=OFFICIAL_SHA256,lora_r=32,lora_alpha=64,
        qwen=str(qwen),official_repo=str(repo))
    model.requires_grad_(False)
    del ckpt,state
    return model.to(device).eval(),report

class OfficialInference:
    def __init__(self,model,device,threshold=.5):
        self.model=model; self.device=device; self.threshold=threshold
        self._key=None; self._features=None; self._mapping=None
    def __call__(self,sample):
        import torch
        import spconv.pytorch as spconv
        key=sample["scene_name"]
        # Native collate and native SpConv backbone are used unchanged.
        if key!=self._key:
            coordinates,features,inverse,shape=self.model.collate([sample["points"]])
            mapping=sample["superpoints"].to(self.device)
            count=int(mapping.max())+1
            x=spconv.SparseConvTensor(features.to(self.device),coordinates.to(self.device).int(),
                                     shape.tolist(),1)
            visual=self.model.extract_feat(x,mapping,inverse.to(self.device),[0,count])
            self._features=visual; self._mapping=mapping; self._key=key
        # No GT answer, target category, entity ID, or masks enter this object.
        qa=[dict(question=sample["question"],answer="",qa_level="functional",
                 qa_key="qa_data",qa_task_type="grounding",qa_keys="single_turn")]
        decoder=self.model.decoder
        result=decoder(self._features,scene_names=[key],qa_data=qa)
        masks=result["masks"][0][0]
        if masks.ndim!=2 or not torch.isfinite(masks).all():
            raise ValueError("Nonfinite or malformed official decoder logits")
        saved=json.loads((Path(decoder.save_pred_qa_dir)/(key+".json")).read_text(encoding="utf-8"))
        answer=saved["functional"]["qa_data"]["grounding"][0]["pred_answer"]
        # Official single-target postprocessing keeps the first predicted phrase;
        # this is fixed before seeing GT, and never chooses best-IoU query.
        raw=masks[0][self._mapping].float().sigmoid() if len(masks) else None
        return dict(answer=answer,mask=(raw>=self.threshold).cpu().numpy() if raw is not None else None,
            mask_official04=(raw>.4).cpu().numpy() if raw is not None else None,
            reason=None,query_count=int(len(masks)))

def evaluate_both(dataset,index,infer):
    prediction=infer(dataset.inference(index))
    truth=dataset.truth(index); row=dataset.rows[index]
    def score(p):
        r=score_prediction(p,truth,row["target_phrase"])
        return dict(r,evaluation_index=index,sample_id=row["sample_id"],source=row["source"],
            level=row["level"],mode="generated_no_fallback")
    return score(prediction),score(dict(prediction,mask=prediction["mask_official04"]))
