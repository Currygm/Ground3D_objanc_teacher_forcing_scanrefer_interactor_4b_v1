"""Ground3D functional single-turn data and GT-isolated evaluation helpers."""
from __future__ import annotations
import hashlib
import json
import os
import re
import uuid
from collections import Counter, OrderedDict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
DATA_ROOT = Path("/mnt/Datasets/Ground3D_Data")
TASKS = {"object_qa":"functional_object_grounding","part_qa":"functional_part_grounding"}
SCHEMA = "ground3d_functional_objanc_v1"

def project_path(path):
    path=Path(path).expanduser().resolve()
    if not path.is_relative_to(PROJECT):
        raise ValueError("Artifacts must remain inside the Reason3D project")
    return path

def atomic_json(path, payload):
    path=project_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)

def convert_qa(qa, *, source,scene,frame,level,source_file,entity_key,qa_index):
    if source not in ("scannet","scannetpp") or level not in TASKS:
        raise ValueError("unsupported_source_or_level")
    if not re.fullmatch(r"[A-Za-z0-9_-]+",scene) or int(frame)<0:
        raise ValueError("unsafe_scene_or_frame")
    question=qa.get("question");answer=qa.get("answer")
    if not isinstance(question,str) or not question.strip() or not isinstance(answer,str):
        raise ValueError("missing_question_or_answer")
    spans=re.findall(r"<p>(.*?)</p>\s*<SEG>",answer,flags=re.S)
    if len(spans)!=1 or answer.count("<p>")!=1 or answer.count("</p>")!=1 or answer.count("<SEG>")!=1:
        raise ValueError("not_exactly_one_grounded_phrase")
    phrase=" ".join(spans[0].split())
    if not phrase or "<" in phrase or ">" in phrase:
        raise ValueError("empty_or_nested_phrase")
    ids=qa.get("object_id")
    if not isinstance(ids,list) or len(ids)!=1 or type(ids[0]) is not int or ids[0]<0:
        raise ValueError("not_exactly_one_target_id")
    return {"schema":SCHEMA,"sample_id":f"ground3d:{source}:{scene}:frame_{frame}:{level}:{entity_key}:{qa_index}",
        "source":source,"scene_id":scene,"frame":int(frame),"level":level,"task":TASKS[level],
        "question":question,"teacher_answer":f"<obj>{phrase}</obj>","target_phrase":phrase,
        "target_ids":ids,"target_label_field":"object_label" if level=="object_qa" else "part_label",
        "original_answer":answer,"original_question":question,"source_file":str(source_file),
        "source_entity_key":str(entity_key),"source_qa_index":int(qa_index),"source_qa_id":qa.get("id")}

def frame_path(row, data_root=DATA_ROOT):
    # Revalidate row paths even when consuming an edited manifest.
    if row["source"] not in ("scannet","scannetpp") or not re.fullmatch(r"[A-Za-z0-9_-]+",row["scene_id"]) or int(row["frame"])<0:
        raise ValueError("unsafe_scene_or_frame")
    root=Path(data_root).resolve()
    path=(root/row["source"]/"processed_data"/row["scene_id"]/f"frame_{int(row['frame'])}").resolve()
    if not path.is_relative_to(root):
        raise ValueError("frame_path_escape")
    return path

def array_path(prefix, field):
    return prefix.with_name(prefix.name+"_"+field+".npy")

def validate_frame(row,data_root=DATA_ROOT,cache=None,*,check_values=True):
    import numpy as np
    prefix=frame_path(row,data_root)
    expected="object_label" if row["level"]=="object_qa" else "part_label"
    if row["target_label_field"]!=expected:
        raise ValueError("label_space_mismatch")
    key=(str(prefix),expected,bool(check_values))
    if cache is not None and key in cache:
        info=cache[key]
    else:
        try:
            vert=np.load(array_path(prefix,"vert"),mmap_mode="r",allow_pickle=False)
            sp=np.load(array_path(prefix,"sp_label"),mmap_mode="r",allow_pickle=False)
            labels=np.load(array_path(prefix,expected),allow_pickle=False)
        except FileNotFoundError as exc:
            raise ValueError("missing_frame_array") from exc
        if vert.ndim!=2 or vert.shape[1]!=6 or not len(vert):
            raise ValueError("invalid_point_shape")
        if check_values and not np.isfinite(vert).all():
            raise ValueError("nonfinite_point_values")
        if sp.size!=len(vert) or labels.size!=len(vert):
            raise ValueError("point_label_length_mismatch")
        if not np.issubdtype(sp.dtype,np.integer) or not np.issubdtype(labels.dtype,np.integer):
            raise ValueError("noninteger_point_labels")
        values,counts=np.unique(labels,return_counts=True)
        info={"points":len(vert),"target_counts":dict(zip(map(int,values),map(int,counts)))}
        if cache is not None:cache[key]=info
    if any(info["target_counts"].get(i,0)==0 for i in row["target_ids"]):
        raise ValueError("target_has_no_points")
    return info

def _prepare_scene(job):
    source,scene,level,data_root=job
    root=Path(data_root)/source;task=TASKS[level];cache={}
    directory=root/"refined_qa_data"/level/scene/"final_json_outputs"
    paths=sorted(directory.glob("*.json"));rows=[];counts=Counter();excluded=Counter();hashes={}
    if not paths:excluded[f"{source}/{level}/missing_scene_qa_directory"]+=1
    for path in paths:
        match=re.fullmatch(r"(\d+)_(?:object|part)_mapping\.json",path.name)
        if not match:
            excluded[f"{source}/{level}/unexpected_filename"]+=1;continue
        raw=path.read_bytes();hashes[str(path)]=hashlib.sha256(raw).hexdigest()
        payload=json.loads(raw)
        for entity_key,entity in payload.items():
            if not isinstance(entity,dict):continue
            qa_rows=entity.get(task,[])
            if not isinstance(qa_rows,list):
                excluded[f"{source}/{level}/invalid_task_container"]+=1;continue
            for qi,qa in enumerate(qa_rows):
                counts[f"{source}/{level}/candidate"]+=1
                try:
                    if not isinstance(qa,dict):raise ValueError("invalid_qa_record")
                    row=convert_qa(qa,source=source,scene=scene,frame=int(match[1]),level=level,
                        source_file=path,entity_key=entity_key,qa_index=qi)
                    validate_frame(row,data_root,cache,check_values=False)
                except (ValueError,TypeError) as exc:
                    excluded[f"{source}/{level}/{str(exc)}"]+=1;continue
                rows.append(row);counts[f"{source}/{level}/retained"]+=1
    return rows,counts,excluded,hashes

def prepare(split="val",sources=("scannet","scannetpp"), *, data_root=DATA_ROOT,output=None,workers=4):
    from concurrent.futures import ThreadPoolExecutor
    if split not in ("train","val"):raise ValueError("Expected train or val")
    if not 1<=workers<=16:raise ValueError("workers must be between 1 and 16")
    if len(set(sources))!=len(sources):raise ValueError("Duplicate sources")
    output=project_path(output or PROJECT/"data"/f"ground3d_functional_{split}")
    output.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for source in sources:
        if source not in ("scannet","scannetpp"):raise ValueError("Unexpected source")
        scene_ids=(Path(data_root)/source/f"part_ground3d_{split}.txt").read_text(encoding="utf-8").split()
        if len(set(scene_ids))!=len(scene_ids):raise ValueError("Duplicate split scenes")
        for scene in scene_ids:
            if not re.fullmatch(r"[A-Za-z0-9_-]+",scene):raise ValueError("Unsafe split scene")
            for level in TASKS:jobs.append((source,scene,level,str(data_root)))
    destination=output/"annotations.jsonl"
    temporary=destination.with_name(destination.name+"."+uuid.uuid4().hex+".tmp")
    counts=Counter();excluded=Counter();file_hashes={};seen=set();frames=set();scenes_seen=set()
    try:
        with temporary.open("w",encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=workers) as pool:
            # map preserves the fixed source/scene/level order, independent of worker completion.
            for index,(rows,cnt,exc,hashes) in enumerate(pool.map(_prepare_scene,jobs)):
                counts.update(cnt);excluded.update(exc);file_hashes.update(hashes)
                for row in rows:
                    if row["sample_id"] in seen:raise ValueError("Duplicate adapted sample")
                    seen.add(row["sample_id"]);frames.add((row["source"],row["scene_id"],row["frame"]))
                    scenes_seen.add((row["source"],row["scene_id"]))
                    handle.write(json.dumps(row,ensure_ascii=False)+"\n")
                if index%10==0 or index+1==len(jobs):
                    print(f"prepare: {index+1}/{len(jobs)} scene/level jobs; {len(seen)} retained",flush=True)
        if not seen:raise ValueError("No eligible functional QA records")
        os.replace(temporary,destination)
    finally:
        temporary.unlink(missing_ok=True)
    report={"schema":SCHEMA,"protocol":"single-turn functional object/part grounding only",
        "split":split,"data_root":str(Path(data_root).resolve()),"annotations":str(destination),
        "samples":len(seen),"frames":len(frames),"scenes":len(scenes_seen),"counts":dict(counts),
        "exclusions":dict(excluded),"source_annotation_sha256":file_hashes,
        "annotation_sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),
        "excluded_tasks":["multi-turn QA","numerical QA","non-functional spatial QA"],
        "gt_policy":"object_qa uses object_label; part_qa uses part_label; no class-name remapping",
        "point_value_validation":"At inference loading; --check-data enables full CPU numeric checks",
        "source_files_modified":False,"preparation_workers":workers}
    atomic_json(output/"manifest.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="source_annotation_sha256"},indent=2),flush=True)
    return report

class FunctionalDataset:
    """Inference __getitem__ never opens GT labels or exposes teacher answers."""
    def __init__(self,rows,*,data_root=DATA_ROOT,cache_dir=None):
        self.rows=rows;self.data_root=Path(data_root);self.cache_dir=project_path(cache_dir) if cache_dir else None
        self._visual=OrderedDict();self._truth=OrderedDict()
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        import numpy as np
        import torch
        from ground3d_migration.core import prepare_geometry
        row=self.rows[index];prefix=frame_path(row,self.data_root);key=str(prefix)
        if key in self._visual:
            cloud,g=self._visual.pop(key);self._visual[key]=(cloud,g)
        else:
            vert=np.load(array_path(prefix,"vert"),allow_pickle=False)
            sp=np.load(array_path(prefix,"sp_label"),allow_pickle=False)
            if vert.ndim!=2 or vert.shape[1]!=6 or not len(vert) or sp.size!=len(vert) or not np.isfinite(vert).all():
                raise ValueError("Malformed inference frame")
            xyz=torch.from_numpy(vert[:,:3].copy()).float()
            rgb=torch.from_numpy(vert[:,3:6].copy()).float()
            # Ground3D vert.npy stores byte-range RGB; no per-frame extrema guessing.
            if bool((rgb<0).any()) or bool((rgb>255).any()):raise ValueError("Expected RGB in [0,255]")
            rgb=rgb/127.5-1.
            cloud=torch.cat([xyz,rgb],dim=1)
            labels=torch.from_numpy(sp.reshape(-1).copy()).long()
            if self.cache_dir:
                from ground3d_migration.geometry import cached_geometry
                g=cached_geometry(xyz,rgb,labels,self.cache_dir)
            else:g=prepare_geometry(xyz,rgb,labels)
            self._visual[key]=(cloud,g)
            while len(self._visual)>2:self._visual.popitem(last=False)
        return {"point_cloud":cloud,"ground_geometry":g,"point_to_superpoint":g["point_to_superpoint"],
            "superpoint_coord":g["superpoint_coord"],"question":row["question"],"sample_id":row["sample_id"]}
    def truth(self,index):
        import numpy as np
        import torch
        row=self.rows[index];field="object_label" if row["level"]=="object_qa" else "part_label"
        if row["target_label_field"]!=field:raise ValueError("label_space_mismatch")
        path=array_path(frame_path(row,self.data_root),field)
        if str(path) not in self._truth:
            self._truth[str(path)]=np.load(path,allow_pickle=False).reshape(-1)
            while len(self._truth)>2:self._truth.popitem(last=False)
        mask=np.isin(self._truth[str(path)],row["target_ids"])
        if not mask.any():raise ValueError("target_has_no_points")
        return torch.from_numpy(mask.copy())

def infer_teacher(model,batch,answer):
    import torch
    from spatiallm_compressed_visual_objanc_full_attention.model.answer_format import parse_answer
    from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_model import mean_sigmoid_scores
    parsed=parse_answer(answer)
    if not parsed.valid:raise ValueError("Invalid adapted teacher answer")
    points=model._encode_points(batch)
    context=model.answer_lm.encode_full_context(batch["point_cloud"],batch["question"],[answer],visuals=[p.compressed for p in points])
    obj=[i for i,e in enumerate(parsed.entities) if e.role=="OBJ"]
    text,positive,valid=model._positive_maps(context,[obj])
    text=[model.text_adapter(x.to(model.text_adapter.weight.dtype)) for x in text]
    output=model.decoder([model.visual_adapter(x.superpoint_hidden.to(model.visual_adapter.weight.dtype)) for x in points],
        text,[x.superpoint_xyz for x in points],context.token_roles)
    positive=positive[:,:,:output["pred_logits"].shape[-1]].to(output["pred_logits"].device)
    if not positive.any():
        from reason3d_no_fallback import failure
        return failure(answer,"obj_span_truncated",format_valid=True)
    scores=mean_sigmoid_scores(output["pred_logits"],positive,valid.to(output["pred_logits"].device))[0,:,0]
    return {"valid":True,"answer":answer,"reason":None,"format_valid":True,"budget_forced":False,
        "selected_query":int(scores.argmax()),"query_scores":scores,"superpoint_logits":output["pred_masks"][0]}

def evaluate_sample(model,dataset,index,mode="generated_no_fallback",*,threshold=.5,predict=None):
    import torch
    sample=dataset[index];batch={k:[v] for k,v in sample.items()}
    if mode=="generated_no_fallback":
        if predict is None:
            from reason3d_no_fallback import infer_generated_no_fallback
            predict=infer_generated_no_fallback
        prediction=predict(model,batch)
    elif mode=="teacher":
        prediction=infer_teacher(model,batch,dataset.rows[index]["teacher_answer"])
    else:raise ValueError("Unsupported evaluation mode")
    # In generated mode all query selection above is finished before any GT access.
    truth=dataset.truth(index)
    row=dataset.rows[index]
    result={"evaluation_index":index,"sample_id":row["sample_id"],"source":row["source"],"level":row["level"],
        "mode":mode,"valid":bool(prediction["valid"]),"generated_answer":prediction["answer"],
        "format_valid":bool(prediction["format_valid"]),"failure_reason":prediction["reason"],
        "selected_query":prediction["selected_query"],"budget_forced":bool(prediction.get("budget_forced",False)),
        "iou":0.,"acc25":0.,"acc50":0.,"intersection":0,"union":int(truth.sum()),"obj_phrase_correct":False}
    if prediction["valid"]:
        masks=prediction["superpoint_logits"];mapping=sample["point_to_superpoint"].to(masks.device,torch.long)
        selected=(masks[prediction["selected_query"]][mapping].sigmoid()>=threshold)
        truth=truth.to(masks.device,torch.bool)
        if selected.shape!=truth.shape:raise ValueError("Prediction/GT point order differs")
        intersection=int((selected&truth).sum());union=int((selected|truth).sum())
        result.update(iou=intersection/max(union,1),intersection=intersection,union=union)
        result["acc25"]=float(result["iou"]>=.25);result["acc50"]=float(result["iou"]>=.5)
        from spatiallm_compressed_visual_objanc_full_attention.model.answer_format import parse_answer,normalize_text
        parsed=parse_answer(prediction["answer"])
        result["obj_phrase_correct"]=parsed.obj is not None and normalize_text(parsed.obj.phrase)==normalize_text(row["target_phrase"])
    return result

def aggregate(rows):
    n=len(rows);den=max(n,1)
    return {"samples":n,"mIoU":sum(r["iou"] for r in rows)/den,
        "gIoU":sum(r.get("intersection",0) for r in rows)/max(sum(r.get("union",0) for r in rows),1),
        "Acc@0.25":sum(r["acc25"] for r in rows)/den,"Acc@0.50":sum(r["acc50"] for r in rows)/den,
        "failure_rate":sum(not r["valid"] for r in rows)/den,
        "failure_reasons":dict(Counter(r["failure_reason"] for r in rows if not r["valid"])),
        "answer_format_valid_rate":sum(r["format_valid"] for r in rows)/den,
        "obj_phrase_accuracy":sum(r["obj_phrase_correct"] for r in rows)/den,
        "fallback_rate":0.,"policy":"GT-free generated query selection; invalid predictions score zero"}

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--split",choices=["train","val"],default="val")
    p.add_argument("--sources",nargs="+",choices=["scannet","scannetpp"],default=["scannet","scannetpp"])
    p.add_argument("--data-root",type=Path,default=DATA_ROOT);p.add_argument("--output",type=Path);p.add_argument("--workers",type=int,default=4)
    a=p.parse_args();prepare(a.split,a.sources,data_root=a.data_root,output=a.output,workers=a.workers)

def summarize(rows, *, expected_indices,modes):
    expected=set(expected_indices)
    for mode in modes:
        indices=[r["evaluation_index"] for r in rows if r["mode"]==mode]
        if len(indices)!=len(expected) or set(indices)!=expected:
            raise ValueError("Distributed evaluation coverage differs")
    def grouped(selected):
        result={mode:aggregate([r for r in selected if r["mode"]==mode]) for mode in modes}
        if "teacher" in result:result["teacher"]["policy"]="GT answer conditioning diagnostic; no GT query matching"
        return result
    return {"metrics":grouped(rows),
        "by_dataset":{source:grouped([r for r in rows if r["source"]==source]) for source in sorted({r["source"] for r in rows})},
        "by_level":{level:grouped([r for r in rows if r["level"]==level]) for level in sorted({r["level"] for r in rows})},
        "by_dataset_and_level":{source:{level:grouped([r for r in rows if r["source"]==source and r["level"]==level])
            for level in sorted({r["level"] for r in rows if r["source"]==source})}
            for source in sorted({r["source"] for r in rows})}}
