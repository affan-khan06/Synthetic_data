"""Generate fresh records from a saved checkpoint without retraining."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from core.data import export_profiles
from core.models import Codec,VAE,Denoiser,generate

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,default=Path(__file__).parent/'demo_run')
    p.add_argument('--rows',type=int,default=1000)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--method',choices=['Latent diffusion','VAE only'],default='Latent diffusion')
    p.add_argument('--output',type=Path,default=Path('runs/fresh_synthetic.csv'))
    args=p.parse_args()
    if not 1<=args.rows<=10000: p.error('--rows must be between 1 and 10000')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    report=json.loads((args.checkpoint/'report.json').read_text(encoding='utf-8'))
    cfg=report['config']
    states=torch.load(args.checkpoint/'weights.pt',map_location='cpu',weights_only=True)
    codec=Codec()
    for k,v in json.loads((args.checkpoint/'codec.json').read_text()).items(): setattr(codec,k,np.array(v))
    vae=VAE(cfg['latent']); vae.load_state_dict(states['vae']); vae.eval()
    denoiser=Denoiser(cfg['latent']);denoiser.load_state_dict(states['denoiser']);denoiser.eval()
    bundle={'codec':codec,'vae':vae,'denoiser':denoiser,'center':states['center'],'scale':states['scale'],'steps':cfg['steps']}
    data=generate(bundle,args.rows,args.seed,args.method)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    export_profiles(data,args.seed).to_csv(args.output,index=False)
    args.output.with_suffix('.provenance.json').write_text(json.dumps({'source':report['provenance'],'method':args.method,'seed':args.seed,'rows':args.rows,'training_dataset_sha256':report['dataset_sha256']},indent=2),encoding='utf-8')
    print(f'Generated {len(data)} rows: {args.output.resolve()}')
    print(report['provenance'])
