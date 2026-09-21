"""Evaluate a frozen experiment on its held-out real test partition, once."""
from pathlib import Path
import argparse,hashlib,json
import pandas as pd
from core.data import CATEGORICAL
from core.pipeline import load
from core.evaluate import utility,fidelity
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=Path('demo_run'));a=p.parse_args()
    out=a.run/'final_test.json'
    if out.exists(): raise SystemExit('Final test report already exists. Keep it; do not repeatedly tune against the test set.')
    r=load(a.run);test=pd.read_csv(a.run/'test.csv',dtype={c:str for c in CATEGORICAL})
    seed=r['report']['config']['seed']
    rows=[{'method':'Real-trained reference',**utility(r['train'],test,seed)}]
    for name,df in r['samples'].items():
        summary,_=fidelity(r['train'],test,df)
        rows.append({'method':name,**summary,**utility(df,test,seed)})
    report={'partition':'real held-out test','test_rows':len(test),'positive_rate':float(test['default'].astype(int).mean()),
            'frozen_config':r['report']['config'],'dataset_sha256':r['report']['dataset_sha256'],
            'checkpoint_sha256':hashlib.sha256((a.run/'weights.pt').read_bytes()).hexdigest(),'metrics':rows,
            'scope':'Single predefined configuration and model seed. No hyperparameter tuning using this test evaluation. UCI Taiwan credit clients; not Indian rural households.'}
    out.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(pd.DataFrame(rows)[['method','pr_auc','f1']].to_string(index=False))
