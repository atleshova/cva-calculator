"""
Streamlit prototype: CVA SCVA calculator (prototype)

Clean fixed version — ready to paste into GitHub.

Key fix: ensure `edited` (from Streamlit editor) is converted into a pandas DataFrame (`edited_df`) before using calculations. This avoids the `DeltaGenerator` error.
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import io

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

# --- Sample data ---
SAMPLE = pd.DataFrame([
    {'TransactionID':'T1','CounterpartyID':'CPTY1','Instrument':'IRS','AssetClass':'rates','Notional':10000000.0,'RemainingMaturity':5.0,'EAD':1709678.88,'MTM':0.0,'Collateral':0.0,'SupervisoryRW':0.03},
    {'TransactionID':'T2','CounterpartyID':'CPTY1','Instrument':'IRS','AssetClass':'rates','Notional':2000000.0,'RemainingMaturity':2.0,'EAD':200000.0,'MTM':0.0,'Collateral':0.0,'SupervisoryRW':0.03},
    {'TransactionID':'T3','CounterpartyID':'CPTY2','Instrument':'FX Swap','AssetClass':'fx','Notional':5000000.0,'RemainingMaturity':3.0,'EAD':500000.0,'MTM':0.0,'Collateral':0.0,'SupervisoryRW':0.10}
])

# --- Sidebar ---
st.sidebar.header('Calculation options')
method = st.sidebar.selectbox('Method', ['SCVA (Art.384)', 'OEM (Art.385)', 'ACVA (placeholder)'])
apply_eu_exemptions = st.sidebar.checkbox('Apply EU exemptions (illustrative)', value=True)

ead_source = st.sidebar.radio('EAD source', ['User-provided', 'Simple SA-CCR estimator', 'Upload SA-CCR CSV'])
split_enabled = st.sidebar.checkbox('Enable 50/50 split', value=True)
split_counterparty = st.sidebar.text_input('Counterparty to split 50/50', value='CPTY1' if split_enabled else '')

st.sidebar.markdown('---')
st.sidebar.header('Regulatory panel')
st.sidebar.markdown(f"**Method used:** {method}")
if method.startswith('SCVA'):
    st.sidebar.markdown('- CRR Art.384: Standardised CVA formula')
elif method.startswith('OEM'):
    st.sidebar.markdown('- CRR Art.385: OEM alternative')
else:
    st.sidebar.markdown('- ACVA requires supervisory approval (not computed here)')

st.sidebar.markdown(f"**EU exemptions:** {'Yes' if apply_eu_exemptions else 'No'}")

st.sidebar.markdown(f"**EAD source:** {ead_source}")

# --- Main area ---
st.title('CVA — Standardised CVA (SCVA) Prototype')
st.write('Upload or edit trades and compute CVA capital (K, RWA).')

uploaded = st.file_uploader('Upload CSV (optional)', type=['csv'])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f'Failed to read CSV: {e}')
        df = SAMPLE.copy()
else:
    df = SAMPLE.copy()

# Editable data
try:
    edited = st.data_editor(df, num_rows='dynamic', use_container_width=True)
except Exception:
    edited = df.copy()
    st.info('Fallback: static dataframe (no editing).')

# Convert editor output to DataFrame
edited_df = pd.DataFrame(edited)

# Apply EAD logic
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
            edited_df['EAD'] = edited_df['EAD_upl'].combine_first(edited_df['EAD'])
            edited_df = edited_df.drop(columns=[c for c in edited_df.columns if c.endswith('_upl')])

# Ensure numeric types
for col in ['RemainingMaturity','EAD','SupervisoryRW','Notional']:
    if col in edited_df.columns:
        edited_df[col] = pd.to_numeric(edited_df[col], errors='coerce').fillna(0.0)

# Compute DF and Term
edited_df['DF'] = edited_df['RemainingMaturity'].apply(lambda m: discount_factor(m))
edited_df['Term'] = edited_df.apply(lambda r: single_counterparty_term(r['SupervisoryRW'], r['RemainingMaturity'], r['EAD'], r['DF']), axis=1)

# Aggregate
agg = edited_df.groupby('CounterpartyID').agg(
    Notional=('Notional','sum'),
    EAD=('EAD','sum'),
    EffectiveMaturity=('RemainingMaturity','max'),
    AvgDF=('DF','mean'),
    SupervisoryRW=('SupervisoryRW','first'),
    TermSum=('Term','sum')).reset_index()
agg['t_i'] = agg['TermSum']

if apply_eu_exemptions:
    agg['is_exempt'] = agg.apply(lambda r: (r['SupervisoryRW']<=0.03 and r['Notional']<=15_000_000), axis=1)
    agg['t_i_used'] = agg.apply(lambda r: 0.0 if r['is_exempt'] else r['t_i'], axis=1)
else:
    agg['is_exempt'] = False
    agg['t_i_used'] = agg['t_i']

terms = list(agg['t_i_used'].astype(float))
K = portfolio_K(terms)
RWA = 12.5*K

st.subheader('Aggregated by counterparty')
st.dataframe(agg)

st.subheader('Portfolio results')
st.metric('K', f"{K:.2f}")
st.metric('RWA', f"{RWA:.2f}")

# Download results
out = io.BytesIO()
with pd.ExcelWriter(out, engine='openpyxl') as writer:
    edited_df.to_excel(writer, sheet_name='Inputs', index=False)
    agg.to_excel(writer, sheet_name='Counterparty_Aggregated', index=False)
    pd.DataFrame([{'Metric':'K','Value':K},{'Metric':'RWA','Value':RWA}]).to_excel(writer, sheet_name='Portfolio_Results', index=False)
out.seek(0)
st.download_button('Download Excel', data=out, file_name='CVA_results.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
