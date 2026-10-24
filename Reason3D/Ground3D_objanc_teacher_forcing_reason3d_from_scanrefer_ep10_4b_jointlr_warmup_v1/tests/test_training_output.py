import json,tempfile,unittest
from pathlib import Path

class FreshOutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/"cache/tmp")
        self.root=Path(self.tmp.name);self.config={"experiment":"reason3d","epochs":13}
    def tearDown(self):self.tmp.cleanup()
    def check(self):
        from training_output import validate_fresh_output
        return validate_fresh_output(self.root,self.config)
    def test_empty_metrics_from_failed_initialization_is_reusable(self):
        (self.root/"metrics.jsonl").touch();(self.root/"config.json").write_text(json.dumps(self.config),encoding="utf-8")
        self.assertEqual(self.check(),"empty_initialization")
        self.assertTrue((self.root/"metrics.jsonl").exists())
    def test_nonempty_metrics_still_blocks_overwrite(self):
        (self.root/"metrics.jsonl").write_text('{"step":10}\n',encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError,"resume"):self.check()
    def test_checkpoint_and_incomplete_checkpoint_both_block(self):
        for name in ["epoch_003_step_00000504.pt","epoch_003_step_00000504.pt.tmp"]:
            f=self.root/name;f.touch()
            with self.assertRaises(FileExistsError):self.check()
            f.unlink()
    def test_different_configuration_does_not_mix_runs(self):
        (self.root/"metrics.jsonl").touch();(self.root/"config.json").write_text('{"epochs":99}',encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError,"config"):self.check()
    def test_new_output_is_accepted(self):
        self.assertEqual(self.check(),"fresh")

    def test_metadata_symlink_cannot_escape_output(self):
        outside=self.root.parent/"outside_metadata_test"
        (self.root/"metrics.jsonl").symlink_to(outside)
        with self.assertRaises(ValueError):self.check()

if __name__=="__main__":unittest.main()
