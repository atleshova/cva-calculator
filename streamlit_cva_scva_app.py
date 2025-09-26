"""
Streamlit prototype: CVA SCVA calculator (prototype)

Updated version with:
- PDF embedding support: place `CRR_575_2013.pdf` in the repo root and the app will preview it.
- Sidebar article links for Articles 381–386. The app parses the PDF (pdfplumber) to locate page numbers for each article dynamically; falls back to approximate pages if parsing fails.
- Right-side PDF preview that opens at the selected article page.
- Robust Excel export fallback: tries to use openpyxl; if openpyxl missing on the host it falls back to CSV download to avoid runtime crashes (addresses the openpyxl/pandas error seen in logs).
- Fixed DeltaGenerator / DataFrame conversion issues (we always coerce the editor output into a pandas DataFrame before calculations).
- Link in the sidebar to your Custom GPT page with an attempt to embed it; if embedding is blocked the link opens in a new tab.

Notes: add `CRR_575_2013.pdf` to your GitHub repo root for the PDF preview to work.

Requirements additions: pdfplumber, openpyxl (if you want Excel download). See requirements.txt in the repo.

"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import io
import os
import base64
from pathlib import Path

# Optional PDF parsing library
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
    rc = 0.0
    alpha = 1.0
    return alpha * (rc + add_on)

# PDF utilities
PDF_FILENAME = 'CRR_575_2013.pdf'  # Put this file in your repo root


def find_article_pages(pdf_path, articles):
    """Return a dict mapping article string -> page number (1-based) found by searching the PDF text.
    If pdfplumber isn't available or search fails for an article, the article won't be mapped.
    """
    mapping = {}
    if not PDFPLUMBER_AVAILABLE:
        return mapping
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or '').lower()
                for art in articles:
                    key = art.lower()
                    if key in text and art not in mapping:
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

# --- Sidebar: Calculation options & Regulatory Panel ---
st.sidebar.header('Calculation options')
method = st.sidebar.selectbox('Method', ['SCVA (Art.384)', 'OEM (Art.385)', 'ACVA (Internal model - placeholder)'])
apply_eu_exemptions = st.sidebar.checkbox('Apply EU exemptions (illustrative)', value=True)

ead_source = st.sidebar.radio('EAD source', ['User-provided', 'Simple SA-CCR estimator', 'Upload SA-CCR CSV'])
split_enabled = st.sidebar.checkbox('Enable 50/50 split', value=True)
split_counterparty = st.sidebar.text_input('Counterparty to split 50/50', value='CPTY1' if split_enabled else '')

st.sidebar.markdown('---')
st.sidebar.header('Regulatory panel (click an article to preview)')
ARTICLES = [f'Article {i}' for i in range(381, 387)]

# show article radio buttons
sel_article = st.sidebar.radio('Open article', ARTICLES)

# show simple metadata
st.sidebar.markdown(f"**Method used:** {method}")
st.sidebar.markdown(f"**EU exemptions applied:** {'Yes' if apply_eu_exemptions else 'No'}")
st.sidebar.markdown(f"**EAD source:** {ead_source}")

st.sidebar.markdown('---')
st.sidebar.markdown('**External tools / links**')
st.sidebar.markdown('- [Open CRR on EUR-Lex](http://data.europa.eu/eli/reg/2013/575/oj/eng)')
# attempt to embed Custom GPT: many hosts block embedding; provide link
CUSTOM_GPT_URL = 'https://chat.openai.com/g/g-68d3fd8f6cb881919a46c5e96e188006-cva-calculation'
st.sidebar.markdown(f"[Open Custom GPT chat]({CUSTOM_GPT_URL})")
try:
    st.sidebar.components.v1.iframe(CUSTOM_GPT_URL, height=300)
except Exception:
    st.sidebar.write('Embedding external sites may be blocked by the host — click the link above to open the Custom GPT in a new tab.')

# --- Main area ---
st.title('CVA — Standardised CVA (SCVA) Prototype')
st.write('Upload or edit trades and compute CVA capital (K, RWA).')

uploaded = st.file_uploader('Upload transactions CSV (optional)', type=['csv'])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f'Failed to read CSV: {e}')
        df = SAMPLE.copy()
else:
    df = SAMPLE.copy()

# Editable data editor (experimental)
try:
    edited = st.data_editor(df, num_rows='dynamic', use_container_width=True)
except Exception:
    edited = df.copy()
    st.info('Editable data editor not available in this Streamlit version; using static table.')

# Convert to DataFrame (safe)
edited_df = pd.DataFrame(edited)

# EAD source handling
if ead_source.startswith('Simple'):
    for i, row in edited_df.iterrows():
        if pd.isna(row.get('EAD', None)) or float(row.get('EAD', 0)) == 0:
            edited_df.at[i, 'EAD'] = simple_sa_ccr_estimate(row.get('Notional', 0), asset_class=row.get('AssetClass', 'rates'))

if ead_source.startswith('Upload'):
    ead_file = st.file_uploader('Upload SA-CCR CSV with TransactionID & EAD', type=['csv'], key='eadupload')
    if ead_file is not None:
        ead_df = pd.read_csv(ead_file)
        if 'TransactionID' in ead_df.columns and 'EAD' in ead_df.columns:
            edited_df = edited_df.merge(ead_df[['TransactionID','EAD']], on='TransactionID', how='left', suffixes=('','_upl'))
            if 'EAD_upl' in edited_df.columns:
                edited_df['EAD'] = edited_df['EAD_upl'].combine_first(edited_df['EAD'])
                edited_df = edited_df.drop(columns=[c for c in edited_df.columns if c.endswith('_upl')])
        else:
            st.error('SA-CCR CSV must contain TransactionID and EAD columns')

# Coerce numeric columns
for col in ['RemainingMaturity','EAD','SupervisoryRW','Notional']:
    if col in edited_df.columns:
        edited_df[col] = pd.to_numeric(edited_df[col], errors='coerce').fillna(0.0)

# Compute DF and Term
edited_df['DF'] = edited_df['RemainingMaturity'].apply(lambda m: discount_factor(float(m)))
edited_df['Term'] = edited_df.apply(lambda r: single_counterparty_term(r['SupervisoryRW'], r['RemainingMaturity'], r['EAD'], r['DF']), axis=1)

# Aggregate per counterparty
agg = edited_df.groupby('CounterpartyID').agg(
    Notional=('Notional','sum'),
    EAD=('EAD','sum'),
    EffectiveMaturity=('RemainingMaturity','max'),
    AvgDF=('DF','mean'),
    SupervisoryRW=('SupervisoryRW','first'),
    TermSum=('Term','sum')).reset_index()
agg['t_i'] = agg['TermSum']

# Apply EU exemptions (illustrative)
if apply_eu_exemptions:
    agg['is_exempt'] = agg.apply(lambda r: (r['SupervisoryRW']<=0.03 and r['Notional']<=15_000_000), axis=1)
    agg['t_i_used'] = agg.apply(lambda r: 0.0 if r['is_exempt'] else r['t_i'], axis=1)
else:
    agg['is_exempt'] = False
    agg['t_i_used'] = agg['t_i']

# Portfolio K
terms = list(agg['t_i_used'].astype(float))
K = portfolio_K(terms)
RWA = 12.5*K

# 50/50 split scenario (if enabled)
split_results = None
if split_enabled and split_counterparty:
    if split_counterparty in edited_df['CounterpartyID'].values:
        mask = edited_df['CounterpartyID'] == split_counterparty
        to_split = edited_df[mask].copy()
        a = to_split.copy(); a['CounterpartyID'] = a['CounterpartyID'] + '_A'; a['EAD'] = a['EAD']*0.5; a['Notional'] = a['Notional']*0.5
        b = to_split.copy(); b['CounterpartyID'] = b['CounterpartyID'] + '_B'; b['EAD'] = b['EAD']*0.5; b['Notional'] = b['Notional']*0.5
        others = edited_df[~mask].copy()
        split_full = pd.concat([others, a, b], ignore_index=True)
        split_full['DF'] = split_full['RemainingMaturity'].apply(lambda m: discount_factor(float(m)))
        split_full['Term'] = split_full.apply(lambda r: single_counterparty_term(r['SupervisoryRW'], r['RemainingMaturity'], r['EAD'], r['DF']), axis=1)
        agg_s = split_full.groupby('CounterpartyID').agg(TermSum=('Term','sum')).reset_index()
        K_split = portfolio_K(list(agg_s['TermSum']), corr=0.25, multiplier=2.33)
        RWA_split = K_split * 12.5
        split_results = {'K_split':K_split, 'RWA_split':RWA_split}

# --- Display results ---
col_left, col_right = st.columns([3,2])
with col_left:
    st.subheader('Aggregated by counterparty')
    st.dataframe(agg)

    st.subheader('Portfolio results')
    st.metric('K', f"{K:.2f}")
    st.metric('RWA', f"{RWA:.2f}")
    if split_results:
        st.subheader('50/50 split scenario')
        st.write(split_results)

    # Export: try Excel first, fallback to CSV if openpyxl not available
    out = io.BytesIO()
    try:
        with pd.ExcelWriter(out, engine='openpyxl') as writer:
            edited_df.to_excel(writer, sheet_name='Inputs', index=False)
            agg.to_excel(writer, sheet_name='Counterparty_Aggregated', index=False)
            pd.DataFrame([{'Metric':'K','Value':K},{'Metric':'RWA','Value':RWA}]).to_excel(writer, sheet_name='Portfolio_Results', index=False)
        out.seek(0)
        st.download_button('Download results (Excel)', data=out, file_name='CVA_results.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    except Exception as e:
        # Fallback: provide CSV downloads
        st.warning('Excel export unavailable on this host — offering CSV exports instead.')
        st.download_button('Download Inputs (CSV)', data=edited_df.to_csv(index=False).encode(), file_name='inputs.csv', mime='text/csv')
        st.download_button('Download Aggregated (CSV)', data=agg.to_csv(index=False).encode(), file_name='aggregated.csv', mime='text/csv')

# --- Right panel: PDF preview for selected article ---
with col_right:
    st.subheader('Regulation preview')
    pdf_path = Path(PDF_FILENAME)
    if pdf_path.exists():
        # find article pages
        mapping = {}
        try:
            mapping = find_article_pages(str(pdf_path), [f'Article {i}' for i in range(381, 387)])
        except Exception:
            mapping = {}

        # If pdfplumber couldn't find pages, provide a fallback mapping (approximate pages in your uploaded PDF)
        fallback = {
            'Article 381': 400,
            'Article 382': 403,
            'Article 383': 406,
            'Article 384': 409,
            'Article 385': 412,
            'Article 386': 416,
        }
        page = mapping.get(sel_article, fallback.get(sel_article, 1))

        # Embed PDF with page anchor (works in many browsers)
        try:
            pdf_url = f"{PDF_FILENAME}#page={page}"
            st.markdown(f"**Showing {sel_article} — page {page}**")
            st.components.v1.iframe(pdf_url, height=800)
        except Exception as e:
            st.error('Unable to embed PDF in this environment. The file exists in the repo but embedding may be restricted.')
            st.markdown(f"[Open PDF (raw)]({PDF_FILENAME})")
    else:
        st.info(f'Put {PDF_FILENAME} in the repository root to enable PDF preview.')

# End of app
