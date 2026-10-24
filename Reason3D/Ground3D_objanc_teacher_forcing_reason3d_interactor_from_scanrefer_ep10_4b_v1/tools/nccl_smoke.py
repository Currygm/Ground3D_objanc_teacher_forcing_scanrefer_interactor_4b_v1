"""No dataset/model weights: test first ALLGATHER, both DDP phases and reductions."""
import json,os,sys,time
from pathlib import Path
from datetime import timedelta
rank=int(os.environ["RANK"]);local=int(os.environ["LOCAL_RANK"]);world=int(os.environ["WORLD_SIZE"])
print(json.dumps({"event":"worker_boot","rank":rank}),flush=True)
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
torch.set_num_threads(1)
torch.cuda.set_device(local)
timeout=timedelta(seconds=int(os.getenv("NCCL_PREFLIGHT_COLLECTIVE_TIMEOUT_SECONDS","45")))
dist.init_process_group("nccl",timeout=timeout)
control=dist.new_group(backend="gloo",timeout=timeout)
dist.monitored_barrier(group=control,timeout=timeout)
start=time.time()
x=torch.tensor([rank],device="cuda",dtype=torch.long)
received=[torch.empty_like(x) for _ in range(world)]
dist.all_gather(received,x)
assert [int(v.item()) for v in received]==list(range(world))
class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.language=nn.Linear(32,8,bias=False);self.visual=nn.Linear(32,8,bias=False);self.joint=False
    def forward(self,x):
        return self.language(x)+(self.visual(x) if self.joint else 0)
model=Tiny().cuda()
phases=[]
for joint in [False,True]:
    model.joint=joint;model.visual.requires_grad_(joint);model.zero_grad(set_to_none=True)
    ddp=DDP(model,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True)
    ddp(torch.ones(2,32,device="cuda")*(rank+1)).sum().backward()
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert torch.allclose(parameter.grad,torch.full_like(parameter,(world+1))),"Gradient averaging differs"
    del ddp
    phases.append("joint" if joint else "answer_only")
value=torch.ones(1024*1024,device="cuda");dist.all_reduce(value);assert value.eq(world).all()
dist.broadcast(value,src=0);torch.cuda.synchronize()
report={"rank":rank,"passed":True,"phases":phases,"world_size":world,
        "seconds":time.time()-start,"visible_devices":os.getenv("CUDA_VISIBLE_DEVICES"),
        "allgather":True,"ddp_backward":True,"allreduce_elements":value.numel(),"broadcast":True}
target=Path(sys.argv[1]).resolve()
project=Path(__file__).resolve().parents[1]
if not target.is_relative_to(project):raise ValueError("Probe output escapes project")
f=target/f"rank_{rank:05d}.json";tmp=f.with_suffix(".tmp")
tmp.write_text(json.dumps(report)+"\n",encoding="utf-8");tmp.replace(f)
print(json.dumps({"rank":rank,"event":"nccl_probe_passed"}),flush=True)
dist.destroy_process_group()
