import torch
from torch.utils.data import DataLoader,TensorDataset
import train_segmentation as runtime

def test_dataloader_seed_does_not_consume_model_rng_at_resume():
    factory=getattr(runtime,"data_loader_generator",None)
    assert callable(factory),"Dedicated loader RNG missing"
    torch.manual_seed(88)
    state=torch.get_rng_state()
    loader=DataLoader(TensorDataset(torch.arange(4)),batch_size=1,generator=factory(42,0,3))
    list(loader)
    assert torch.equal(torch.get_rng_state(),state)
    expected=torch.rand(4)
    torch.set_rng_state(state)
    resumed=DataLoader(TensorDataset(torch.arange(4)),batch_size=1,generator=factory(42,0,3))
    list(resumed)
    assert torch.equal(torch.rand(4),expected)
    assert not torch.equal(factory(42,0,3).get_state(),factory(42,1,3).get_state())
