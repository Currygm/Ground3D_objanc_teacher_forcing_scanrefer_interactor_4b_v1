"""Bounded, disposable NCCL probes and an environment inherited by training."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time,shlex
from pathlib import Path

SOCKET_ENV={
    "NCCL_NET":"Socket","NCCL_SOCKET_IFNAME":"lo","GLOO_SOCKET_IFNAME":"lo",
    "NCCL_IB_DISABLE":"1","NCCL_P2P_DISABLE":"1","NCCL_SHM_DISABLE":"1",
    "NCCL_CUMEM_ENABLE":"0",
}

def select_profile(run,base_env,requested="auto"):
    if requested not in ("auto","native","socket"):raise ValueError("Unknown NCCL profile")
    profiles=("native","socket") if requested=="auto" else (requested,)
    for name in profiles:
        env=dict(base_env)
        if name=="socket":env.update(SOCKET_ENV)
        if run(name,env):return name,env
    raise RuntimeError("NCCL preflight failed; training was not started. Inspect per-profile logs.")

def run_bounded(command,*,env,log,timeout):
    with Path(log).open("w",encoding="utf-8") as out:
        proc=subprocess.Popen(command,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        timed_out=False
        try:returncode=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;returncode=124
        if timed_out or returncode:
            # This process group was created solely for this probe, never reused training.
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait()
        return returncode

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--world-size",type=int,required=True)
    parser.add_argument("--env-file",type=Path,required=True)
    parser.add_argument("--profile",choices=["auto","native","socket"],default="auto")
    parser.add_argument("--timeout",type=float,default=float(os.getenv("NCCL_PREFLIGHT_TIMEOUT_SECONDS","240")))
    args=parser.parse_args()
    project=Path(__file__).resolve().parent
    env_file=args.env_file.resolve()
    if not env_file.is_relative_to(project):raise ValueError("Preflight output escapes project")
    if args.world_size<2:raise ValueError("NCCL preflight requires at least two GPUs")
    base=dict(os.environ);base.setdefault("NCCL_DEBUG","INFO")
    base["PYTHONUNBUFFERED"]="1"
    report_dir=env_file.with_suffix("")
    report_dir.mkdir(parents=True,exist_ok=False)
    attempts=[]
    def probe(name,env):
        target=report_dir/name;target.mkdir()
        log=target/"probe.log"
        command=[sys.executable,"-u","-B","-m","torch.distributed.run","--standalone",
                 "--nproc_per_node="+str(args.world_size),str(project/"tools/nccl_smoke.py"),str(target)]
        print(json.dumps({"event":"nccl_preflight_start","profile":name,"log":str(log),
                          "visible_devices":env.get("CUDA_VISIBLE_DEVICES")}),flush=True)
        code=run_bounded(command,env=env,log=log,timeout=args.timeout)
        reports=[json.loads(f.read_text(encoding="utf-8")) for f in target.glob("rank_*.json")]
        passed=code==0 and {x["rank"] for x in reports}==set(range(args.world_size)) and all(x["passed"] for x in reports)
        attempts.append({"profile":name,"exit_code":code,"passed":passed,"reports":reports,"log":str(log)})
        print(json.dumps({"event":"nccl_preflight_result","profile":name,"passed":passed,"exit_code":code}),flush=True)
        return passed
    try:name,chosen=select_profile(probe,base,args.profile)
    except Exception:
        (report_dir/"summary.json").write_text(json.dumps({"attempts":attempts,"training_started":False},indent=2)+"\n",encoding="utf-8")
        raise
    lines=["# Verified communication environment; generated before model initialization.",
           "export NCCL_PROFILE_SELECTED="+shlex.quote(name)]
    if name=="socket":
        lines.extend("export "+k+"="+shlex.quote(chosen[k]) for k in SOCKET_ENV)
    env_file.write_text("\n".join(lines)+"\n",encoding="utf-8")
    (report_dir/"summary.json").write_text(json.dumps({"selected_profile":name,"attempts":attempts,
         "visible_devices":chosen.get("CUDA_VISIBLE_DEVICES"),"transport":{k:chosen.get(k) for k in SOCKET_ENV}},indent=2)+"\n",encoding="utf-8")
    print("NCCL preflight passed; selected "+name+"; environment: "+str(env_file),flush=True)

if __name__=="__main__":main()
