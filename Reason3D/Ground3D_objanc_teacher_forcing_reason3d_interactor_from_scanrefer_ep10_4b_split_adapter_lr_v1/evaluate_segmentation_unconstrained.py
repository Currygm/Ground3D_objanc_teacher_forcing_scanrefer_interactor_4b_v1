from reason3d_unconstrained import GENERATION_PROTOCOL
from types import MethodType
from ground3d_migration.unconstrained import generate_one_unconstrained
def install_generation(model,sink):
    model.answer_lm.generate_one=MethodType(generate_one_unconstrained,model.answer_lm)
    model.answer_lm._raw_generation_sink=sink
import json
from pathlib import Path
def finalize_summary(output,world_size):
    path=Path(output)/"summary.json"
    summary=json.loads(path.read_text(encoding="utf-8"))
    summary["generation_protocol"]=dict(GENERATION_PROTOCOL)
    summary["generation_protocol"]["format_failure_policy"]="existing_zero_obj_decoder"
    path.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
