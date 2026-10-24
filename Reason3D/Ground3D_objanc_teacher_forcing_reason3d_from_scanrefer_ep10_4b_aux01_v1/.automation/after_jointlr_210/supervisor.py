from pathlib import Path
import os,sys,json,subprocess,signal
from datetime import datetime,timezone
signal.signal(signal.SIGHUP,signal.SIG_IGN)
project,started,exited=map(Path,sys.argv[1:])
def write(path,value):
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
    os.replace(tmp,path)
command=["/bin/bash",str(project/"scripts/train_then_eval.sh")]
child=subprocess.Popen(command,cwd=project,stdin=subprocess.DEVNULL)
write(started,{"supervisor_pid":os.getpid(),"child_pid":child.pid,"command":command,"started_utc":datetime.now(timezone.utc).isoformat(),"gpu_ids":os.environ["GPU_IDS"]})
print("SUPERVISOR_CHILD",child.pid,flush=True)
code=child.wait()
write(exited,{"exit_code":code,"finished_utc":datetime.now(timezone.utc).isoformat(),"supervisor_pid":os.getpid(),"child_pid":child.pid})
print("SUPERVISOR_EXIT",code,flush=True)
sys.exit(code if code>=0 else 128-code)
