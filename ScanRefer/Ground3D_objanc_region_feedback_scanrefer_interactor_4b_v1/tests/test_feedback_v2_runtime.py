import inspect
import pytest
from torch.utils.data import Dataset
from region_feedback.cache import CacheWriter,CachedDataset
from test_feedback_v2_cache import record

def test_cached_loader_uses_full_train_and_exact_eval_shards(tmp_path):
    from region_feedback.runtime_v2 import cached_loader
    records=[record() for _ in range(5)]
    for split in ["train","val"]:
        writer=CacheWriter(tmp_path/split,split,{},5)
        for i,r in enumerate(records):
            r=dict(r,index=i,sample_id=f"sample:{i}");writer.write(i,r)
        writer.finalize()
    cfg={"cache_dir":str(tmp_path),"training":{"batch_size":1,"num_workers":0,"seed":42}}
    ds=CachedDataset(tmp_path/"val",{},expected_count=5,split="val")
    shards=[]
    for rank in range(3):
        subset,loader=cached_loader(ds,cfg,"val",rank,3)
        shards.append([r["index"] for b in loader for r in b])
    assert shards==[[0,3],[1,4],[2]]
    train=CachedDataset(tmp_path/"train",{},expected_count=5,split="train")
    _,loader=cached_loader(train,cfg,"train",0,1,epoch=2)
    assert sorted(r["index"] for b in loader for r in b)==list(range(5))

def test_cache_budget_estimate_and_failure():
    from region_feedback.cache_runtime import check_disk_budget
    assert check_disk_budget(1024,10,1,10**9)==10240
    with pytest.raises(ValueError,match="budget"):check_disk_budget(1024**3,2,1,10**12)
    with pytest.raises(ValueError,match="space"):check_disk_budget(1024**3,2,10,1)

def test_cache_training_never_invokes_online_proposals():
    from region_feedback import runtime_v2
    text=inspect.getsource(runtime_v2.train_main)
    assert ".first_stage.predict" not in text and "build_first_stage" not in text
    assert "cached" in text.lower()
