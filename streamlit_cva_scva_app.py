"""
Streamlit prototype: CVA SCVA calculator with PDF preview (Base64 embed)

Changes in this version:
- PDF embedding now uses Base64 encoding, so the regulation PDF is always displayed correctly (fixes the issue where the app showed itself instead of the PDF).
- Place `CRR_575_2013.pdf` in the repo root. On app launch, it will be read and embedded into an iframe.
- Sidebar article links (Art. 381–386) still supported. We use pdfplumber to detect pages if possible, otherwise fallback to approximate page numbers.

"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import io
import base64
from pathlib import Path

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except Exception:
    PDFPLUMBER_AVAILABLE = False

st.set_page_config(page_title='CVA SCVA Prototype', layout='wide')

# --- Helper functions ---

def discount_factor(M, r=0.05):
    if M <= 0:
        return 1.0
    return (1 - math.exp(-r * M)) / (r * M)

def single_counterparty_term(w, M, EAD, DF):
    return float(w) * float(M) * float(EAD) * float(DF)

def portfolio_K(counterparty_terms, corr=0.25, multiplier=2.33):
    t = counterparty_terms
    sum_sq = sum(x * x for x in t)
    sum_t = sum(t)
    cross = corr * (sum_t * sum_t - sum_sq)
    variance = sum_sq + cross
    variance = max(variance, 0.0)
    return multiplier * math.sqrt(variance)

def simple_sa_ccr_estimate(notional, asset_class='rates'):
    sf_map = {'rates': 0.005, 'fx': 0.01, 'credit': 0.03, 'equity': 0.06}
    sf = sf_map.get(asset_class, 0.01)
    add_on = abs(float(notional)) * sf
    return 1.0 * add_on

# PDF utilities
PDF_FILENAME = 'CRR_575_2013.pdf'

def find_article_pages(pdf_path, articles):
    mapping = {}
    if not PDFPLUMBER_AVAILABLE:
        return mapping
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or '').lower()
                for art in articles:
                    if art.lower() in text and art not in mapping:
                        mapping[art] = i
        return mapping
    except Exception:
        return {}

# --- Sample data ---
SAMPLE = pd.DataFrame([
    {'TransactionID':'T1','CounterpartyID':'CPTY1','Instrument':'IRS','AssetClass':'rates','Notional':10000000.0,'RemainingMaturity':5.0,'EAD':1709678.88,'MTM':0.0,'Collateral':0.0,'SupervisoryRW':0.03},
    {'TransactionID':'T2','CounterpartyID':'CPTY1','Instrument':'IRS','AssetClass':'rates','Notional':2000000.0,'RemainingMaturity':2.0,'EAD':200000.0,'MTM':0.0,'Collateral':0.0,'SupervisoryRW':0.03},
    {'TransactionID':'T3','CounterpartyID':'CPTY2','Instrument':'FX Swap','AssetClass':'fx','Notional':5000000.0,'RemainingMaturity':3.0,'EAD':500000.0,'MTM':0.0,'Collateral':0.0,'SupervisoryRW':0.10}
])

# --- Sidebar ---
st.sidebar.header('Calculation options')
method = st.sidebar.selectbox('Method', ['SCVA (Art.384)', 'OEM (Art.385)', 'ACVA (Internal model - placeholder)'])
apply_eu_exemptions = st.sidebar.checkbox('Apply EU exemptions (illustrative)', value=True)
ead_source = st.sidebar.radio('EAD source', ['User-provided', 'Simple SA-CCR estimator', 'Upload SA-CCR CSV'])
split_enabled = st.sidebar.checkbox('Enable 50/50 split', value=True)
split_counterparty = st.sidebar.text_input('Counterparty to split 50/50', value='CPTY1' if split_enabled else '')

st.sidebar.markdown('---')
st.sidebar.header('Regulatory panel (click an article to preview)')
ARTICLES = [f'Article {i}' for i in range(381, 387)]
sel_article = st.sidebar.radio('Open article', ARTICLES)

st.sidebar.markdown(f"**Method used:** {method}")
st.sidebar.markdown(f"**EU exemptions applied:** {'Yes' if apply_eu_exemptions else 'No'}")
st.sidebar.markdown(f"**EAD source:** {ead_source}")

# --- Main ---
st.title('CVA — Standardised CVA (SCVA) Prototype')

uploaded = st.file_uploader('Upload transactions CSV (optional)', type=['csv'])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f'Failed to read CSV: {e}')
        df = SAMPLE.copy()
else:
    df = SAMPLE.copy()

try:
    edited = st.data_editor(df, num_rows='dynamic', use_container_width=True)
except Exception:
    edited = df.copy()
edited_df = pd.DataFrame(edited)

# Coerce numerics
for col in ['RemainingMaturity','EAD','SupervisoryRW','Notional']:
    if col in edited_df.columns:
        edited_df[col] = pd.to_numeric(edited_df[col], errors='coerce').fillna(0.0)

# Fill EAD if needed
if ead_source.startswith('Simple'):
    for i, row in edited_df.iterrows():
        if row['EAD'] == 0:
            edited_df.at[i,'EAD'] = simple_sa_ccr_estimate(row['Notional'], asset_class=row['AssetClass'])

# DF and Terms
edited_df['DF'] = edited_df['RemainingMaturity'].apply(lambda m: discount_factor(float(m)))
edited_df['Term'] = edited_df.apply(lambda r: single_counterparty_term(r['SupervisoryRW'], r['RemainingMaturity'], r['EAD'], r['DF']), axis=1)

# Aggregate
agg = edited_df.groupby('CounterpartyID').agg(Notional=('Notional','sum'),EAD=('EAD','sum'),EffectiveMaturity=('RemainingMaturity','max'),AvgDF=('DF','mean'),SupervisoryRW=('SupervisoryRW','first'),TermSum=('Term','sum')).reset_index()
agg['t_i'] = agg['TermSum']
if apply_eu_exemptions:
    agg['is_exempt'] = agg.apply(lambda r: (r['SupervisoryRW']<=0.03 and r['Notional']<=15_000_000), axis=1)
    agg['t_i_used'] = agg.apply(lambda r: 0.0 if r['is_exempt'] else r['t_i'], axis=1)
else:
    agg['is_exempt'] = False
    agg['t_i_used'] = agg['t_i']

K = portfolio_K(list(agg['t_i_used']))
RWA = 12.5*K

# --- Display ---
col_left, col_right = st.columns([3,2])
with col_left:
    st.subheader('Aggregated by counterparty')
    st.dataframe(agg)
    st.subheader('Portfolio results')
    st.metric('K', f"{K:.2f}")
    st.metric('RWA', f"{RWA:.2f}")

with col_right:
    st.subheader('Regulation preview')
    pdf_path = Path(PDF_FILENAME)
    if pdf_path.exists():
        mapping = find_article_pages(str(pdf_path), ARTICLES)
        fallback = {'Article 381':400,'Article 382':403,'Article 383':406,'Article 384':409,'Article 385':412,'Article 386':416}
        page = mapping.get(sel_article, fallback.get(sel_article, 1))
        try:
            with open(pdf_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page}" width="100%" height="800" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
        except Exception as e:
            st.error('Could not load PDF preview.')
    else:
        st.info(f'Put {PDF_FILENAME} in the repo root to enable preview.')
