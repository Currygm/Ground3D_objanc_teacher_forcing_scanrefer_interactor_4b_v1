"""Independent unconstrained evaluation; native training/evaluation stay unchanged.

Usage: python evaluate_segmentation_unconstrained.py CONFIG CHECKPOINT OUTPUT [native options]
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from types import MethodType
import evaluate_segmentation as native
from ground3d_migration.unconstrained import generate_one_unconstrained, merge_prompt_visual


def install_generation(model, sink):
    model.answer_lm.generate_one=MethodType(generate_one_unconstrained,model.answer_lm)
    model.answer_lm._raw_generation_sink=sink


class UnconstrainedObjAncSegmentationModel(native.ObjAncSegmentationModel):
    def __init__(self, config):
        super().__init__(config)
        self._raw_sample={}
        rank=int(os.environ.get("RANK","0"))
        path=Path(os.environ["GROUND3D_UNCONSTRAINED_OUTPUT"])/f"raw_generations.rank_{rank:05d}.jsonl"
        # Exclusive creation prevents mixing this run with previous generated text.
        path.parent.mkdir(parents=True,exist_ok=True)
        self._raw_trace=path.open("x",encoding="utf-8")
        self._raw_count=0
        install_generation(self,self._record_generation)
        self.answer_lm._merge=MethodType(merge_prompt_visual,self.answer_lm)

    def _record_generation(self, row):
        self._raw_trace.write(json.dumps({**self._raw_sample,**row},ensure_ascii=False)+"\n")
        self._raw_trace.flush()
        self._raw_count+=1
        if self._raw_count==1 or self._raw_count%25==0:
            print(f"[rank {os.environ.get('RANK','0')}] unconstrained generations: {self._raw_count}",flush=True)

    def _choose_answers(self,batch,point_rows,answer_mode):
        if answer_mode=="generated":
            if len(batch["question"])!=1:
                raise ValueError("Unconstrained raw tracing requires evaluation batch size 1")
            self._raw_sample={
                "sample_id":str(batch["sample_id"][0]),
                "evaluation_index":int(batch["evaluation_index"][0]),
                "question":batch["question"][0],
            }
        # Preserve native format-failure -> zero OBJ and strictRaw -> zero-score policy.
        return super()._choose_answers(batch,point_rows,answer_mode)


def finalize_summary(output,world_size):
    output=Path(output)
    path=output/"summary.json"
    summary=json.loads(path.read_text(encoding="utf-8"))
    summary["generation_protocol"]={
        "format_constraints":False,"greedy":True,
        "special_token_suppression":False,"undefined_id_suppression":False,
        "forced_tag_closure":False,"forced_eos":False,
        "phrase_token_limit":None,"total_token_limit_source":"model.generation.max_new_tokens",
        "format_failure_policy":"existing_zero_obj_decoder",
        "strict_raw_failure_policy":"zero_obj_metrics_in_full_sample_denominator",
        "answer_repair":False,"teacher_fallback":False,
        "decoder_context":"native causal replay of original decoded answer; native synthetic OBJ on format failure",
        "raw_files":[str(output/f"raw_generations.rank_{rank:05d}.jsonl") for rank in range(world_size)],
    }
    temporary=path.with_suffix(".json.unconstrained.tmp")
    temporary.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    temporary.replace(path)
    from runtime_helpers import sha256
    (output/'completion.json').write_text(json.dumps({'validation_samples':summary['validation_samples'],'summary_sha256':sha256(path),'generation_protocol':'unconstrained_zero_obj_v1'},indent=2)+'\n')


def main():
    if len(sys.argv)<4 or any(x in sys.argv for x in ("-h","--help")):
        native.main()
        return
    config=native.load_segmentation_config(Path(sys.argv[1]))
    output=native.ensure_within(Path(sys.argv[3]),config["project"]["artifact_root"],must_exist=False)
    if (output/"summary.json").exists():
        raise FileExistsError(f"Use a new output directory; summary already exists: {output}")
    # Resolve the actual output before any raw files are created.
    os.environ["GROUND3D_UNCONSTRAINED_OUTPUT"]=str(output)
    native.ObjAncSegmentationModel=UnconstrainedObjAncSegmentationModel
    native.main()  # Keeps original @torch.no_grad(), checkpoint validation and rank aggregation.
    if int(os.environ.get("RANK","0"))==0:
        finalize_summary(output,int(os.environ.get("WORLD_SIZE","1")))
        print(f"Unconstrained evaluation complete: {output/'summary.json'}",flush=True)


if __name__=="__main__":
    main()
