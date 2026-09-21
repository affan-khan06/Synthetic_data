"""Retrieve UCI's actual workbook, verify its structure, and preserve provenance."""
from pathlib import Path
from urllib.request import urlopen, Request
from zipfile import ZipFile
from io import BytesIO
import hashlib,json
import pandas as pd
from core.data import validate
ROOT=Path(__file__).resolve().parent
URL='https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip'
if __name__=='__main__':
    folder=ROOT/'data';folder.mkdir(exist_ok=True)
    archive=folder/'uci350_original.zip'
    if not archive.exists():
        with urlopen(Request(URL,headers={'User-Agent':'BorrowerData research prototype'}),timeout=90) as r: archive.write_bytes(r.read())
    with ZipFile(archive) as z:
        filename=next(n for n in z.namelist() if n.lower().endswith('.xls'))
        raw=z.read(filename)
    df=pd.read_excel(BytesIO(raw),header=1,engine='xlrd')
    if len(df)!=30000: raise ValueError('Unexpected source row count; review source before proceeding.')
    df=validate(df)
    output=folder/'uci_credit_clients.csv';df.to_csv(output,index=False)
    manifest={'dataset':'Default of Credit Card Clients','creator':'I-Cheng Yeh','citation':'Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H.',
              'source_page':'https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients','download_url':URL,'license':'CC BY 4.0','license_url':'https://creativecommons.org/licenses/by/4.0/',
              'source_rows':len(df),'source_features':23,'currency':'New Taiwan dollar (NT$)','period':'April–September 2005 history; next-month default target','population':'Taiwan credit-card clients',
              'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'workbook_sha256':hashlib.sha256(raw).hexdigest(),'csv_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
              'transformations':['Read original XLS with second row as header','Rename target to default','Preserve original numeric category codes, including undocumented codes','Exclude source ID from modeling','Drop exact feature+target duplicate records before partitioning'],
              'category_notes':'UCI summary does not define every observed code. EDUCATION 0/5/6, MARRIAGE 0 and repayment -2/0 are retained without inventing definitions.'}
    (folder/'provenance.json').write_text(json.dumps(manifest,indent=2))
    print('Loaded actual UCI workbook:',df.shape,'Default proportion:',df['default'].astype(int).mean())
