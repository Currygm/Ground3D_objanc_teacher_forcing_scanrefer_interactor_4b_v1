import ast
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
class EntryContract(unittest.TestCase):
    def test_separate_evaluator_uses_eval_model_and_runtime_signature(self):
        tree=ast.parse((ROOT/"evaluate_reason3d_unconstrained.py").read_text(encoding="utf-8"))
        imports=[n for n in tree.body if isinstance(n,ast.ImportFrom)]
        self.assertTrue(any(n.module=="reason3d_unconstrained_model" and any(x.name=="ObjAncSegmentationModel" for x in n.names) for n in imports))
        self.assertTrue(any(n.module=="reason3d_runtime" and any(x.name=="OBJECTIVE" for x in n.names) for n in imports))
    def test_eval_subclass_keeps_original_parameter_and_signature_methods(self):
        tree=ast.parse((ROOT/"reason3d_unconstrained_model.py").read_text(encoding="utf-8"))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="ObjAncSegmentationModel")
        names={n.name for n in cls.body if isinstance(n,ast.FunctionDef)}
        self.assertFalse(names & {"architecture_signature","trainable_state_dict","_encode_points","_positive_maps","__init__"})
    def test_script_invokes_independent_entry(self):
        text=(ROOT/"scripts/evaluate_unconstrained.sh").read_text(encoding="utf-8")
        self.assertIn("evaluate_reason3d_unconstrained.py",text)
        self.assertIn("runtime.sh",text)
        self.assertNotIn("--max-samples",text)
if __name__=="__main__":unittest.main()
