import unittest, math, importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class ScheduleContract(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location("joint_lr_schedule",ROOT/"joint_lr_schedule.py")
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        self.t={"epochs":13,"answer_warmup_epochs":3,"joint_epochs":10,
            "cosine_min_lr_ratio":.1,"joint_warmup_steps":100,
            "joint_qwen_lora_lr":5e-6,"joint_structured_token_lr":5e-6,
            "joint_visual_lr":3e-6,"joint_adapter_lr":3e-5,"joint_decoder_lr":3e-5,"qwen_lora_lr":1e-5,"structured_token_lr":1e-5,"visual_lr":1e-5,"adapter_lr":1e-4,"decoder_lr":1e-4}
    def lr(self,name,step):
        return self.module.learning_rate(name,step,168,self.t)
    def test_answer_stage_exactly_matches_original_continuous_cosine(self):
        for step in (0,1,100,503):
            ratio=.1+.9*.5*(1+math.cos(math.pi*step/2184))
            for name,base in [("qwen_lora",1e-5),("structured_tokens",1e-5),("ground_visual",1e-5),("adapters",1e-4),("decoder",1e-4)]:
                self.assertAlmostEqual(self.lr(name,step),base*ratio,places=15)
    def test_joint_boundary_language_retains_target_and_new_groups_warm_up(self):
        self.assertAlmostEqual(self.lr("qwen_lora",504),5e-6)
        self.assertAlmostEqual(self.lr("structured_tokens",504),5e-6)
        self.assertAlmostEqual(self.lr("ground_visual",504),3e-8)
        self.assertAlmostEqual(self.lr("adapters",504),3e-7)
        self.assertAlmostEqual(self.lr("decoder",504),3e-7)
        self.assertAlmostEqual(self.lr("decoder",603),3e-5)
        self.assertAlmostEqual(self.lr("decoder",604),3e-5)
    def test_final_minimum_and_decay_monotonic(self):
        for name,target in [("qwen_lora",5e-6),("structured_tokens",5e-6),("ground_visual",3e-6),("adapters",3e-5),("decoder",3e-5)]:
            self.assertAlmostEqual(self.lr(name,2184),target*.1)
            rates=[self.lr(name,x) for x in range(604,2185)]
            self.assertTrue(all(a>=b for a,b in zip(rates,rates[1:])))
    def test_bad_schedule_rejected(self):
        for name in ("typo","interactor"):
            with self.assertRaises(ValueError):self.lr(name,504)
        self.t["joint_warmup_steps"]=1680
        with self.assertRaises(ValueError):self.lr("decoder",504)
if __name__=="__main__":unittest.main()
