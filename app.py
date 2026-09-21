from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import json
import hashlib
import html
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from core.data import load_source, validate, NUMERIC, CATEGORICAL, MONTHS, MONTH_LABELS, COLUMNS, PROVENANCE, export_profiles, display_profiles
from core.pipeline import run, load, load_bundle, DEFAULT_CONFIG, FILES

from core.models import generate

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Synthetic Borrower Data',page_icon='◈',layout='wide')
st.markdown('''<style>
.block-container{padding-top:2rem;max-width:1450px}
h1,h2,h3{letter-spacing:-.035em;color:#43382e}
[data-testid="stAppViewContainer"], [data-testid="stHeader"]{background:#f5f0e7}
[data-testid="stMetric"]{background:#fffcf6;border:1px solid #ded2bf;border-radius:15px;padding:20px;border-top:4px solid #a4774d}
[data-testid="stSidebar"]{background:#eae0d1;border-right:1px solid #d8c8b3}
.hero{background:linear-gradient(115deg,#e9dcc8,#f1e8da 68%,#e3d4be);border:1px solid #d6c3a8;border-radius:20px;padding:30px 36px;color:#43382e;margin-bottom:22px}
.hero h1{color:#43382e;margin:5px 0;font-size:38px}
.hero p{color:#665747;margin:4px 0}
.eyebrow{font-size:12px;letter-spacing:.18em;color:#795734;font-weight:700}
.pill{display:inline-block;border:1px solid #c5ad8c;background:#f8f2e7;border-radius:20px;padding:5px 12px;margin-top:14px;font-size:12px;color:#644b34}
[data-baseweb="tab-list"]{border-bottom:1px solid #d8c8b3}
[data-baseweb="tab"][aria-selected="true"]{color:#805934}
[data-testid="stAlertContainer"]{border-radius:12px}
[data-testid="stAlertContainer"][data-baseweb="notification"]{border:1px solid #d8c8b3}
</style>''',unsafe_allow_html=True)
st.markdown('''<div class="hero"><div class="eyebrow">REAL DATA / SYNTHETIC BORROWERS</div><h1>New records. Measurable evidence.</h1><p>Learn from real UCI credit clients. Generate new borrower profiles with latent diffusion.</p><span class="pill">MID-REVIEW PROTOTYPE</span> <span class="pill">CPU TRAINING</span> <span class="pill">VALIDATION RESULTS</span></div>''',unsafe_allow_html=True)

with st.sidebar:
    st.title('Borrower Data')
    st.caption('Experiment controls')
    source=st.radio('Data source',['Real UCI credit clients (30,000)','Upload UCI-schema CSV'])
    upload=None
    if source.startswith('Upload'):
        upload=st.file_uploader('CSV matching the finance schema',type=['csv'])
        st.caption('One row per credit client. Use the documented UCI fields. Source IDs are excluded from learning.')
    st.divider()
    st.subheader('Training configuration')
    preset=st.selectbox('Training budget',['Presentation (40 + 60 epochs)','Quick smoke run (5 + 8 epochs)','Extended (80 + 120 epochs)'])
    latent=st.select_slider('Latent dimensions',options=[4,8,12,16,24],value=12)
    kl=st.select_slider('KL regularization',options=[.005,.01,.03,.1],value=.03)
    n=st.select_slider('Generated rows per method',options=[1000,3000,6000,12000,18000],value=18000)
    seed=st.number_input('Random seed',0,9999,42)
    train_clicked=st.button('Train & evaluate',type='primary',width='stretch')
    reset_clicked=st.button('Restore included experiment',width='stretch')
    st.caption('All displayed metrics come from saved or live computations. Training runs locally; allow it to finish.')

@st.cache_data
def bundled(): return load(ROOT/'demo_run')

if 'result' not in st.session_state:
    st.session_state.result=bundled()
if 'experiments' not in st.session_state: st.session_state.experiments=[]
if reset_clicked:
    st.session_state.result=bundled()
    st.session_state.experiments=[]
if train_clicked:
    try:
        if source.startswith('Upload'):
            if upload is None: raise ValueError('Choose a CSV file before training.')
            raw=pd.read_csv(upload)
            provenance='User-uploaded CSV. Source and representativeness not independently verified.'
        else:
            raw=load_source()
            provenance=PROVENANCE
        data=validate(raw)
        ve,de={'Presentation (40 + 60 epochs)':(40,60),'Quick smoke run (5 + 8 epochs)':(5,8),'Extended (80 + 120 epochs)':(80,120)}[preset]
        cfg={**DEFAULT_CONFIG,'vae_epochs':ve,'diffusion_epochs':de,'latent':latent,'kl_weight':kl,'synthetic_rows':n,'seed':int(seed)}
        bar=st.progress(0,text='Preparing training split')
        result=run(data,cfg,provenance,lambda f,s:bar.progress(min(f,1.),text=s))
        st.session_state.result=result
        record=result['report']
        st.session_state.experiments.append({'run':len(st.session_state.experiments)+1,'dataset':record['dataset_sha256'],'seed':int(seed),'rows':n,'latent':latent,'kl':kl,'metrics':record['metrics']})
        st.success(f"Training and evaluation finished in {record['elapsed_seconds']:.1f} seconds on this machine.")
    except Exception as exc:
        st.error(f'Training did not finish: {exc}. The last completed experiment remains visible below.')

result=st.session_state.result
report=result['report']; reference=result['validation']; training=result['train']; samples=result['samples']
st.info(report['provenance'])
st.caption('Viewing the last completed experiment. Sidebar changes take effect after Train & evaluate. These panels use validation records. The packaged final test report is separate and never used for training.')
a,b,c,d=st.columns(4)
a.metric('Real training clients',f"{report['split_counts']['train']:,}")
b.metric('Synthetic rows / method',f"{report['config']['synthetic_rows']:,}")
c.metric('Real validation clients',f"{report['split_counts']['validation']:,}")
d.metric('Held-out test clients',f"{report['split_counts']['test_reserved']:,}")
method=st.selectbox('Inspect generator',list(samples),index=3)
synthetic=samples[method]; metrics=pd.DataFrame(report['metrics'])
selected=metrics.set_index('method').loc[method]
tabs=st.tabs(['Overview','Borrower profiles','Fidelity','Utility','Privacy screen','Training & export','Real dataset & test'])
COLORS={'Reference validation':'#514238','Generated':'#a77442','Independent':'#aaa08d','Gaussian copula':'#b68a46','VAE only':'#817156','Latent diffusion':'#a77442'}
def chart(fig):
    fig.update_layout(template='plotly_white', colorway=['#a77442','#817156','#514238','#b68a46','#8b6b5e'], paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color='#514238'),margin=dict(l=15,r=15,t=45,b=15))
    fig.update_xaxes(gridcolor='#e4dacb', zerolinecolor='#d1bfa7')
    fig.update_yaxes(gridcolor='#e4dacb', zerolinecolor='#d1bfa7')
    st.plotly_chart(fig,width='stretch',theme=None)

with tabs[0]:
    left,right=st.columns([1.5,1])
    with left:
        st.subheader('Does synthesis preserve the pattern?')
        seasonal=pd.DataFrame({'Month':MONTH_LABELS*2,'Mean bill balance (NT$)':list(reference[MONTHS].mean())+list(synthetic[MONTHS].mean()),'Dataset':['Reference validation']*6+['Generated']*6})
        chart(px.line(seasonal,x='Month',y='Mean bill balance (NT$)',color='Dataset',markers=True,color_discrete_map=COLORS,title='Mean credit-card bill balances, April–September 2005'))
    with right:
        st.subheader('Selected model')
        st.metric('Mean marginal error ↓',f"{selected['marginal_error']:.3f}")
        st.metric('Synthetic-trained PR-AUC ↑',f"{selected['pr_auc']:.3f}" if pd.notna(selected['pr_auc']) else 'Unavailable')
        st.caption('Marginal error averages numeric KS statistics and categorical total variation across columns. PR-AUC uses average precision on the source validation set.')
    st.subheader('Working pipeline')
    st.markdown('**1. Split real clients** → **2. Fit training-only preprocessing** → **3. Train mixed-type VAE** → **4. Learn latent denoising** → **5. Decode new records** → **6. Evaluate held-out validation data**')
    st.warning('The source is real Taiwanese credit-card data. Results do not establish performance for Indian rural borrowers or guarantee privacy. Names, income, village and occupation are not present in this dataset.')

with tabs[1]:
    st.subheader('Inspect source and generated records')
    view=st.radio('Table',['Generated','Source training'],horizontal=True)
    shown=training if view=='Source training' else synthetic
    st.dataframe(display_profiles(export_profiles(shown,report['config']['seed'])) .head(200) if view=='Generated' else display_profiles(shown.head(200)),width='stretch',hide_index=True)
    st.caption(f'{len(shown):,} rows total. Preview limited to 200 rows. No original client IDs enter model training. Table labels are expanded for readability; CSV preserves documented codes.')
    st.download_button('Download generated CSV',export_profiles(synthetic,report['config']['seed']).to_csv(index=False),f'synthetic_borrowers_{FILES[method]}.csv','text/csv',key='data_csv')

    st.subheader('Generate a fresh CSV without retraining')
    fresh_rows=st.number_input('Fresh profiles',100,10000,1000,100)
    fresh_seed=st.number_input('Sampling seed',0,999999,123)
    if st.button('Generate fresh borrower profiles'):
        with st.spinner('Sampling from trained model...'):
            bundle=result.get('bundle') or load_bundle(ROOT/'demo_run')
            selected_method=method if method in ['VAE only','Latent diffusion'] else 'Latent diffusion'
            fresh=generate(bundle,int(fresh_rows),int(fresh_seed),selected_method)
            st.session_state.fresh={'data':export_profiles(fresh,int(fresh_seed)),'method':selected_method,'seed':int(fresh_seed),'source':report['provenance']}
    if 'fresh' in st.session_state:
        fresh=st.session_state.fresh
        st.caption(f"Last fresh batch: {fresh['method']}, seed {fresh['seed']}. This batch has not been evaluated; charts describe the evaluated experiment.")
        st.dataframe(display_profiles(fresh['data'].head(10)),hide_index=True,width='stretch')
        st.download_button('Download fresh borrower CSV',fresh['data'].to_csv(index=False),'synthetic_borrowers_fresh.csv','text/csv')
        st.download_button('Download fresh batch provenance',json.dumps({k:v for k,v in fresh.items() if k!='data'},indent=2),'fresh_provenance.json','application/json')

    st.subheader('Individual generated bill histories')
    profiles=synthetic[MONTHS].head(5).T.copy(); profiles.index=MONTH_LABELS; profiles.columns=[f'Profile {i+1}' for i in range(5)]
    chart(px.line(profiles,color_discrete_sequence=['#a77442','#817156','#514238','#b68a46','#8b6b5e'],labels={'index':'Month','value':'Bill balance (NT$)','variable':'Synthetic client'},markers=True))
    st.caption('Each row represents a synthetic client with six historical bill balances. These are generated historical features, not forecasts. Negative bills in the source are retained; no interpretation beyond the dataset documentation is assumed.')

with tabs[2]:
    st.subheader('Distribution fidelity')
    col=st.selectbox('Compare column',COLUMNS)
    plot=pd.concat([reference[[col]].assign(Dataset='Reference validation'),synthetic[[col]].assign(Dataset='Generated')])
    chart(px.histogram(plot,x=col,color='Dataset',histnorm='probability',barmode='overlay' if col in NUMERIC else 'group',opacity=.7,color_discrete_map=COLORS,nbins=35))
    st.dataframe(pd.DataFrame(report['per_column'][method]).round(4),hide_index=True,width='stretch')
    a,b=st.columns(2)
    with a:
        chart(px.imshow(reference[NUMERIC].corr(method='spearman').fillna(0),zmin=-1,zmax=1,color_continuous_scale=[[0,'#6e7c68'],[.5,'#fbf6ec'],[1,'#9b603e']],title='Reference numeric correlations'))
    with b:
        chart(px.imshow(synthetic[NUMERIC].corr(method='spearman').fillna(0),zmin=-1,zmax=1,color_continuous_scale=[[0,'#6e7c68'],[.5,'#fbf6ec'],[1,'#9b603e']],title='Generated numeric correlations'))
    st.subheader('PCA projection')
    scaler=StandardScaler().fit(training[NUMERIC]); pca=PCA(n_components=2).fit(scaler.transform(training[NUMERIC]))
    frames=[]
    for label,df in [('Reference validation',reference),('Generated',synthetic)]:
        coords=pca.transform(scaler.transform(df[NUMERIC]))
        frames.append(pd.DataFrame({'PC1':coords[:,0],'PC2':coords[:,1],'Dataset':label}))
    chart(px.scatter(pd.concat(frames),x='PC1',y='PC2',color='Dataset',opacity=.4,color_discrete_map=COLORS))
    st.caption('PCA and scaling are fitted only on training numeric features. Visual overlap is descriptive, not proof of fidelity. KS p-values are intentionally omitted.')

with tabs[3]:
    st.subheader('Can generated records train a useful classifier?')
    st.caption('Identical balanced logistic regression protocol. Default = 1 is the positive class. Threshold = 0.5 for F1 and balanced accuracy. Each classifier fits its own preprocessing on its training records.')
    baseline=report['source_baseline']
    table=pd.concat([pd.DataFrame([{'method':'Source-trained reference',**baseline}]),metrics],ignore_index=True)
    chart(px.bar(table,x='method',y='pr_auc',color='method',color_discrete_map=COLORS,range_y=[0,1],labels={'pr_auc':'PR-AUC (average precision)','method':''}))
    st.caption(f"Random-ranking reference: validation default prevalence ≈ {report['validation_default_rate']:.3f}. Synthetic-trained classifiers are evaluated on held-out real validation clients (TSTR). The reference classifier is trained on real training clients (TRTR).")
    st.dataframe(table[['method','pr_auc','f1','balanced_accuracy','roc_auc','utility_status']].round(4),hide_index=True,width='stretch')
    st.subheader('Fidelity versus utility')
    chart(px.scatter(metrics,x='marginal_error',y='pr_auc',color='method',text='method',color_discrete_map=COLORS,labels={'marginal_error':'Marginal error ↓','pr_auc':'PR-AUC ↑'}))
    st.caption('This compares four methods at one configuration. It is not an optimized Pareto frontier. A stronger model is closer to the upper-left corner.')
    matching=[e for e in st.session_state.experiments if e['dataset']==report['dataset_sha256'] and e['seed']==report['config']['seed'] and e['rows']==report['config']['synthetic_rows']]
    if len(matching)>1:
        st.subheader('Live configuration comparisons')
        rows=[]
        for e in matching:
            m=next(x for x in e['metrics'] if x['method']=='Latent diffusion')
            rows.append({'run':e['run'],'latent':e['latent'],'kl':e['kl'],'marginal_error':m['marginal_error'],'pr_auc':m['pr_auc']})
        st.dataframe(pd.DataFrame(rows),hide_index=True)
        st.caption('Only runs with the same dataset hash, seed and generated row count appear. These are validation experiments; no test-set model selection.')

with tabs[4]:
    st.subheader('Memorization screening')
    p=report['privacy'][method]
    a,b,c=st.columns(3)
    a.metric('Exact training duplicates',str(p['exact_training_duplicates']))
    b.metric('Median distance to train',f"{p['dcr_train_median']:.4f}")
    c.metric('Train / validation distance',f"{p['dcr_ratio']:.3f}")
    distances=pd.DataFrame({'Distance':p['train_distances']+p['validation_distances'],'Reference':['Training']*p['reference_rows']+['Validation']*p['reference_rows']})
    chart(px.histogram(distances,x='Distance',color='Reference',barmode='overlay',opacity=.65,nbins=30,color_discrete_sequence=['#817156','#a77442']))
    st.caption(f"Distances use {p['reference_rows']} generated rows and equally sized source reference subsets. Each numeric feature uses training-range-scaled absolute distance; categories contribute 0/1 mismatch. The score averages over all columns, including the outcome. Exact-duplicate checks use all training and generated rows.")
    st.warning('No differential privacy guarantee. A ratio near 1 or zero duplicates does not establish privacy. These measurements are only basic proximity screens.')

with tabs[5]:
    st.subheader('Training evidence')
    hist=pd.DataFrame(report['history'])
    a,b=st.columns(2)
    with a:
        vae=hist[hist.stage=='VAE']
        chart(px.line(vae,x='epoch',y=['train_loss','validation_reconstruction'],title='VAE optimization',color_discrete_sequence=['#a77442','#817156']))
        st.caption('Training loss = reconstruction + KL penalty; validation line = deterministic reconstruction only. These are different objectives.')
    with b:
        chart(px.line(hist[hist.stage=='Diffusion'],x='epoch',y='train_loss',title='Diffusion noise-prediction MSE',color_discrete_sequence=['#b68a46']))
        st.caption('Diffusion loss is measured on training latents. A decreasing loss alone does not prove generation quality.')
    st.json({'config':report['config'],'elapsed_seconds':report['elapsed_seconds'],'dataset_sha256':report['dataset_sha256'],'partition':report['evaluation_partition']},expanded=False)
    st.download_button('Download experiment report',json.dumps(report,indent=2), 'synthetic_report.json','application/json')
    st.download_button('Download metric table',metrics.to_csv(index=False),'synthetic_metrics.csv','text/csv')
    memory=BytesIO()
    with ZipFile(memory,'w') as z:
        z.writestr('report.json',json.dumps(report,indent=2))
        z.writestr('metrics.csv',metrics.to_csv(index=False))
        for name,df in samples.items(): z.writestr(FILES[name]+'.csv',export_profiles(df,report['config']['seed']).to_csv(index=False))
    st.download_button('Download complete experiment',memory.getvalue(),'synthetic_experiment.zip','application/zip')
    st.subheader('What can be claimed at mid-review')
    st.markdown('- Implemented: mixed-type VAE, latent DDPM, four generators, validation metrics, CSV export.\n- Demonstrated: real UCI client data used for training; synthetic profiles evaluated on held-out real clients.\n- Pending: multiple seeds, CTGAN, robust privacy attacks, and validation in an Indian rural-lending setting.\n- Method: a small TabSyn-inspired implementation; not a reproduction of FinDiff or TabDDPM.')
    with st.expander('Research references and deck corrections'):
        st.markdown('''[TabSyn](https://arxiv.org/abs/2310.09656) motivates latent diffusion. [FinDiff](https://arxiv.org/abs/2309.01472) and [TabDDPM](https://arxiv.org/abs/2209.15421) provide mixed-type diffusion context.

The rebuilt pipeline uses UCI dataset 350, credited to Yeh (2009), under CC BY 4.0. It models actual Taiwan credit-card fields. It does not contain rural Indian incomes or establish differential privacy.''')
st.caption('Synthetic Borrower Data • Research prototype • No lending decisions • Real UCI source / synthetic outputs')

with tabs[6]:
    st.subheader('Verified real-world source')
    manifest=json.loads((ROOT/'data/provenance.json').read_text())
    st.markdown('**Default of Credit Card Clients — UCI, Yeh (2009)**. 30,000 clients; Taiwan; 23 input features; next-month default target; monetary values in NT$.')
    st.markdown('[Official dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)')
    st.write('Personal attributes: age, recorded sex, education and marital status. Financial attributes: credit limit, six repayment-status codes, six bill amounts and six payment amounts.')
    st.warning('No names, addresses, occupation or income fields exist in this source. Undocumented category codes are preserved, not assigned invented meanings. Synthetic IDs are assigned after generation.')
    st.json(manifest,expanded=False)
    st.write(f"Exact duplicate feature/target rows removed before splitting: {report.get('duplicate_rows_removed',0)}. Source IDs never enter generator or classifier inputs.")
    st.subheader('Packaged final hold-out test evaluation')
    final_path=ROOT/'demo_run/final_test.json'
    if final_path.exists():
        final=json.loads(final_path.read_text())
        st.caption('This report applies only to the packaged default checkpoint, not new live experiments. Configuration was fixed before this test evaluation. Do not tune using these test results.')
        st.dataframe(pd.DataFrame(final['metrics']).round(4),hide_index=True,width='stretch')
        st.download_button('Download final test report',json.dumps(final,indent=2),'final_test.json','application/json')
    else:
        st.info('Final test report not available. Use final_test.py after freezing your configuration.')
