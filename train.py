from pathlib import Path
import argparse
import pandas as pd
from core.data import load_source,PROVENANCE
from core.pipeline import run,save
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path);p.add_argument('--output',type=Path,default=Path('demo_run'))
    p.add_argument('--vae-epochs',type=int,default=40);p.add_argument('--diffusion-epochs',type=int,default=60);p.add_argument('--rows',type=int,default=18000)
    a=p.parse_args()
    df=pd.read_csv(a.csv) if a.csv else load_source()
    provenance='User-provided UCI-schema CSV; origin not independently verified.' if a.csv else PROVENANCE
    result=run(df,{'vae_epochs':a.vae_epochs,'diffusion_epochs':a.diffusion_epochs,'synthetic_rows':a.rows},provenance,lambda f,s:print(f'{f:.0%} {s}',flush=True))
    save(result,a.output)
    print(pd.DataFrame(result['report']['metrics'])[['method','marginal_error','pr_auc']].to_string(index=False))
