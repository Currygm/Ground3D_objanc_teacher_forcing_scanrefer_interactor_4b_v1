import unittest,importlib.util,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class AuxContract(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location("aux_loss_contract",ROOT/"aux_loss_contract.py")
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)
    def test_only_aux_weight_may_change(self):
        source={"span_weight":6.,"mask_bce_weight":5.,"dice_weight":2.,"aux_weight":.5}
        new={**source,"aux_weight":.1}
        self.m.validate_loss_config(new,source)
        for field,value in [("span_weight",7.),("aux_weight",.5),("extra",1.)]:
            with self.assertRaises(ValueError):self.m.validate_loss_config({**new,field:value},source)
        self.assertEqual(source["aux_weight"],.5)
    def test_parent_signature_normalization_is_local_and_nonmutating(self):
        signature={"objective":"parent","loss":{"aux_weight":.1,"span_weight":6.},"decoder":{"dim":256}}
        before=copy.deepcopy(signature)
        loaded=self.m.parent_signature_for_loading(signature)
        self.assertEqual(loaded,{**signature,"loss":{"aux_weight":.5,"span_weight":6.}})
        self.assertEqual(signature,before)
        self.assertIsNot(loaded["loss"],signature["loss"])
    def test_normalization_cannot_hide_other_signature_differences(self):
        target={"loss":{"aux_weight":.1,"span_weight":6.},"decoder":{"dim":256}}
        self.assertNotEqual(self.m.parent_signature_for_loading(target),{"loss":{"aux_weight":.5,"span_weight":7.},"decoder":{"dim":256}})
        with self.assertRaises(ValueError):self.m.parent_signature_for_loading({"loss":{"aux_weight":.5}})
if __name__=="__main__":unittest.main()
