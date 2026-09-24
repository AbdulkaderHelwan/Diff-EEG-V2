import sys, time, numpy as np, torch
B="/home/abdulh/scratch/benchmarking2"
sys.path.insert(0, B+"/EEGDM_ref"); sys.path.insert(0, B+"/code")
_o=torch.load; torch.load=lambda *a,**k:_o(*a,**{**k,"weights_only":False})
import model.s4standalone as s4
print("  has_pykeops:", s4.has_pykeops, " has_cuda_extension:", s4.has_cuda_extension, flush=True)
from model.classifier import LatentActivityExtractor, LatentActivityReducer
from model.diffusion_model_pl import PLDiffusionModel
from src.util import staged_mu_law
import data_adapter as A
dev="cuda"
R=dict(query=["gate"],reduce=["std"],rescale=False,L=1000,window_size=200,window_step=200,pool_merge="share",multi_query_merge="seq")
dm=PLDiffusionModel.load_from_checkpoint(B+"/EEGDM_ref/checkpoint/pretrain/backbone.ckpt",map_location=dev)
net=torch.nn.Sequential(LatentActivityExtractor(model=dm.ema.ema_model,diffusion_t=1,query=R["query"],use_cond=None),
                        LatentActivityReducer(**R)).to(dev).eval()
idx=A.build_index("tuab5s","test")[:512]
for bs in (16,32,64,128):
    try:
        X,_=A.materialise(idx[:bs],"tuab5s","eegdm")
        X=np.stack([staged_mu_law(x.copy()) for x in X])
        t=torch.from_numpy(X).float().to(dev)
        with torch.no_grad(): net((t,None)); torch.cuda.synchronize()
        t0=time.time()
        with torch.no_grad():
            for _ in range(3): net((t,None))
        torch.cuda.synchronize()
        dt=(time.time()-t0)/3
        print(f"  batch {bs:>4}: {dt:.3f}s -> {bs/dt:>8.1f} samples/s = {bs/dt*60:>9.0f} /min",flush=True)
    except torch.OutOfMemoryError:
        print(f"  batch {bs:>4}: OOM",flush=True); torch.cuda.empty_cache()
