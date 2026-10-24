"""Regression for the real loop's AMP/DDP context nesting."""
import ast,contextlib,copy,os,subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest,torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("should_step",[False,True])
def test_real_microstep_backward_exits_autocast_but_preserves_no_sync(should_step):
    tree=ast.parse((ROOT/"train_reason3d.py").read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
    block=next(n for n in ast.walk(main) if isinstance(n,ast.Try)
               and any(isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute)
                       and c.func.attr=="backward" for c in ast.walk(n))
               and any(isinstance(c,ast.ExceptHandler) for c in n.handlers)
               and not any(isinstance(c,ast.For) for c in n.body))
    body=copy.deepcopy(block.body)
    # Exercise the real nesting on CPU; production only enables AMP on CUDA.
    for node in ast.walk(ast.Module(body=body,type_ignores=[])):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=="autocast":
            for kw in node.keywords:
                if kw.arg=="enabled":kw.value=ast.Constant(True)
    flags={"sync_active":False};events=[]
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__();self.linear=nn.Linear(8,4)
        @contextlib.contextmanager
        def no_sync(self):
            flags["sync_active"]=True
            try:yield
            finally:flags["sync_active"]=False
        def forward(self,batch,**kwargs):
            out=self.linear(batch)
            events.append(("forward",torch.is_autocast_enabled("cpu"),out.dtype,flags["sync_active"]))
            return {"loss":out.square().mean()}
    model=Tiny()
    model.linear.weight.register_hook(lambda g:events.append(("backward",torch.is_autocast_enabled("cpu"),g.dtype,flags["sync_active"])))
    sync=model.no_sync() if not should_step else contextlib.nullcontext()
    scope=dict(torch=torch,contextlib=contextlib,sync=sync,model=model,
        batch=torch.randn(2,8),epoch=0,step=0,steps_per_epoch=168,
        window=2,device=torch.device("cpu"),t={"amp":True})
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),"<real_microstep>","exec"),scope)
    assert events[0][0:3]==("forward",True,torch.bfloat16)
    assert events[1][0:2]==("backward",False)
    assert events[1][3] is (not should_step)
    assert torch.isfinite(model.linear.weight.grad).all()
    assert model.linear.weight.grad.abs().sum()>0

def test_allocator_default_precedes_torch_import_and_preserves_user_override():
    text=(ROOT/"scripts/runtime.sh").read_text()
    line=next((l for l in text.splitlines() if l.startswith("export PYTORCH_CUDA_ALLOC_CONF=")),None)
    assert line is not None,"Launcher needs fragmentation-resistant allocator default"
    assert text.index(line)<text.index('GPU_UUIDS="$(')
    def result(value):
        env=dict(os.environ)
        if value is None:env.pop("PYTORCH_CUDA_ALLOC_CONF",None)
        else:env["PYTORCH_CUDA_ALLOC_CONF"]=value
        return subprocess.check_output(["bash","-c",line+'\nprintf "%s" "$PYTORCH_CUDA_ALLOC_CONF"'],env=env,text=True)
    assert result(None)=="expandable_segments:True"
    assert result("max_split_size_mb:128")=="max_split_size_mb:128"
