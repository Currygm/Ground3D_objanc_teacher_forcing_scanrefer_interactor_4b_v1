"""Prepare full-sentence functional Ground3D train/val; legacy files stay unchanged."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import ground3d_functional as base
SCHEMA = "ground3d_functional_full_sentence_objanc_v1"
def convert(qa, **metadata):
    row = base.convert_qa(qa, **metadata)
    answer, count = re.subn(r"<p>(.*?)</p>\s*<SEG>", lambda m: "<obj>" + m.group(1) + "</obj>", qa["answer"], flags=re.S)
    if count != 1 or answer.count("<obj>") != 1 or answer.count("</obj>") != 1 or any(t in answer for t in ("<p>", "</p>", "<SEG>")):
        raise ValueError("invalid_formatted_answer")
    row.update(schema=SCHEMA, answer=answer, teacher_answer=answer, phrase_only_answer="<obj>"+row["target_phrase"]+"</obj>")
    return row

def scene_job(job):
    source, scene, level = job
    paths = sorted((base.DATA_ROOT/source/"refined_qa_data"/level/scene/"final_json_outputs").glob("*.json"))
    rows, counts, excluded, hashes, cache = [], Counter(), Counter(), {}, {}
    group = source+"/"+level
    if not paths: excluded[group+"/missing_scene_qa_directory"] += 1
    for path in paths:
        match = re.fullmatch(r"(\d+)_(?:object|part)_mapping\.json", path.name)
        if not match:
            excluded[group+"/unexpected_filename"] += 1
            continue
        raw = path.read_bytes(); hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        for key, entity in json.loads(raw).items():
            if not isinstance(entity, dict): continue
            qas = entity.get(base.TASKS[level], [])
            if not isinstance(qas, list):
                excluded[group+"/invalid_task_container"] += 1
                continue
            for qi, qa in enumerate(qas):
                counts[group+"/candidate"] += 1
                try:
                    if not isinstance(qa, dict): raise ValueError("invalid_qa_record")
                    row = convert(qa, source=source, scene=scene, frame=int(match[1]), level=level, source_file=path, entity_key=key, qa_index=qi)
                    if os.environ.get("GROUND3D_CHECK_FRAMES") == "1":
                        base.validate_frame(row, cache=cache, check_values=False)
                except (ValueError, TypeError) as exc:
                    excluded[group+"/"+str(exc)] += 1
                    continue
                rows.append(row); counts[group+"/retained"] += 1
    return rows, counts, excluded, hashes

def prepare(split, root, workers, scene_lists):
    jobs = [(source,scene,level) for source,scenes in scene_lists.items() for scene in scenes for level in base.TASKS]
    output = base.project_path(root/split); output.mkdir(parents=True, exist_ok=False)
    destination = output/"annotations.jsonl"; temporary = output/"annotations.jsonl.tmp"
    counts, excluded, hashes = Counter(), Counter(), {}
    seen, frames, scenes = set(), set(), set()
    with temporary.open("x", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        for index,(rows,cnt,exc,sha) in enumerate(pool.map(scene_job,jobs)):
            counts.update(cnt); excluded.update(exc); hashes.update(sha)
            for row in rows:
                if row["sample_id"] in seen: raise ValueError("duplicate_sample")
                seen.add(row["sample_id"]); frames.add((row["source"],row["scene_id"],row["frame"])); scenes.add((row["source"],row["scene_id"]))
                handle.write(json.dumps(row,ensure_ascii=False)+"\n")
            if index%20==0 or index+1==len(jobs): print(f"{split}: {index+1}/{len(jobs)} jobs; {len(seen)} samples",flush=True)
    if not seen: raise ValueError("no_samples")
    os.replace(temporary,destination)
    report = dict(schema=SCHEMA,split=split,samples=len(seen),frames=len(frames),scenes=len(scenes),counts=dict(counts),exclusions=dict(excluded),annotations=str(destination),data_root=str(base.DATA_ROOT),annotation_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),source_annotation_sha256=hashes,source_files_modified=False,point_arrays_checked=os.environ.get("GROUND3D_CHECK_FRAMES")=="1",answer_format="Full original answer; replace <p>phrase</p><SEG> with <obj>phrase</obj>; preserve surrounding text",tasks=base.TASKS,excluded_tasks=["multi-turn QA","numerical QA","non-functional spatial QA"],gt_policy="object_qa: object_label; part_qa: part_label; unchanged target IDs")
    base.atomic_json(output/"manifest.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="source_annotation_sha256"}),flush=True)
    return report

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",type=Path,default=base.PROJECT/"data"/"ground3d_functional_full_sentence_v1")
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.workers<=16: parser.error("workers must be in [1,16]")
    root=base.project_path(args.output_root)
    if any((root/split).exists() for split in ("train","val")): raise FileExistsError("Refusing to overwrite existing splits")
    splits={}
    for split in ("train","val"):
        splits[split]={}
        for source in ("scannet","scannetpp"):
            scenes=(base.DATA_ROOT/source/("part_ground3d_"+split+".txt")).read_text().split()
            if len(scenes)!=len(set(scenes)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+",s) for s in scenes): raise ValueError("invalid_split_scenes")
            splits[split][source]=scenes
    for source in ("scannet","scannetpp"):
        if set(splits["train"][source]) & set(splits["val"][source]): raise ValueError("official_split_overlap")
    for split in ("train","val"): prepare(split,root,args.workers,splits[split])
