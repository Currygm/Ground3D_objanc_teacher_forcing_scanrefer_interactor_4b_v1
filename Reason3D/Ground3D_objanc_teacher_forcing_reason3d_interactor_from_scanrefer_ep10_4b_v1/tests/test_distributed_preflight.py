import unittest
from pathlib import Path
import tempfile,os,signal,sys,time

class PreflightTests(unittest.TestCase):
    def test_native_success_does_not_disable_transports(self):
        from distributed_preflight import select_profile
        calls=[]
        name,env=select_profile(lambda n,e:calls.append((n,e)) or True,{"CUDA_VISIBLE_DEVICES":"GPU-a,GPU-b"})
        self.assertEqual(name,"native");self.assertEqual(len(calls),1);self.assertNotIn("NCCL_P2P_DISABLE",env)
    def test_native_failure_retries_socket_only_and_preserves_gpu_mapping(self):
        from distributed_preflight import select_profile
        calls=[]
        def run(name,env):calls.append((name,env));return name=="socket"
        name,env=select_profile(run,{"CUDA_VISIBLE_DEVICES":"GPU-a,GPU-b"})
        self.assertEqual([x[0] for x in calls],["native","socket"])
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"],"GPU-a,GPU-b")
        for key in ["NCCL_P2P_DISABLE","NCCL_SHM_DISABLE","NCCL_IB_DISABLE"]:self.assertEqual(env[key],"1")
        self.assertEqual(env["NCCL_SOCKET_IFNAME"],"lo")
    def test_both_fail_never_returns_training_environment(self):
        from distributed_preflight import select_profile
        with self.assertRaises(RuntimeError):select_profile(lambda *a:False,{})
    def test_explicit_socket_skips_native_probe(self):
        from distributed_preflight import select_profile
        seen=[]
        name,env=select_profile(lambda n,e:seen.append(n) or True,{},requested="socket")
        self.assertEqual(seen,["socket"])
    def test_timeouts_terminate_owned_process_group(self):
        from distributed_preflight import run_bounded
        root=Path(__file__).resolve().parents[1]/"cache/tmp"
        with tempfile.TemporaryDirectory(dir=root) as d:
            log=Path(d)/"probe.log"
            code=run_bounded([sys.executable,"-u","-c","import os,time;print(os.getpid(),flush=True);time.sleep(30)"],env=dict(os.environ),log=log,timeout=.2)
            self.assertEqual(code,124)
            pid=int(log.read_text(encoding="utf-8").splitlines()[0])
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)

if __name__=="__main__":unittest.main()
