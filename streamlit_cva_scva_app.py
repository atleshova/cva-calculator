"""
Streamlit prototype: CVA SCVA calculator (prototype)

This file is a self-contained Streamlit app prototype that:
- Lets you upload or edit a table of trades (TransactionID, CounterpartyID, Instrument, AssetClass, Notional, RemainingMaturity, EAD, MTM, SupervisoryRW)
- Supports selecting method: SCVA (standardised), OEM (Original Exposure Method), ACVA (internal model - placeholder)
- Allows choosing EAD source: user-provided EAD, simplified SA-CCR estimator, or upload SA-CCR CSV
- Toggle to apply/remove EU exemptions (illustrative heuristic)
- Optional 50/50 split of a chosen counterparty
- Shows portfolio K and RWA computed per CRR Art.384 SCVA formula
- Provides a dynamic Regulatory Panel (sidebar) that updates to show which CRR articles / EBA/ACPR docs were used/selected
- Lets you download the results as Excel (values) or CSV

DISCLAIMER: This is a prototype for exploration and illustration only. Use production SA-CCR engines and validated systems for regulatory reporting.

Requirements (create a venv and install):
    pip install streamlit pandas numpy openpyxl

Run locally:
    streamlit run Streamlit_CVA_SCVA_App.py

Quick deploy options (short):
- Streamlit Community Cloud: push this file + requirements.txt to a GitHub repo, then connect the repo in https://share.streamlit.io and deploy.
- Google Cloud Run: create a Dockerfile (example in the README block below) and deploy to Cloud Run. See the detailed README inside this file (near the end).

Files in your dataset referenced by the app's Regulatory Panel (you provided these in the canvas):
- regulation 573 2013 (CRR) CELEX_32013R0575_EN.pdf
- regulation 575-2013 modified version 2025 CRR CELEX_02013R0575-20240709_EN_TXT.pdf
- 2024 03 25_revue_acpr_paquet_bancaire.pdf
- 2024 12 30_Notice_CRD4_marques_revision.pdf
- EBA Report on CVA.pdf
- FRTB-CVA-UNE-REVISION-EN-PROFONDEUR-DE-LA-MESURE-DU-RISQUE-DE-CVA-Juin-2017.pdf
- calcul de la CVA - memoire d actuariat.pdf

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

# --- UI: top-level description ---
st.title('CVA — Standardised CVA (SCVA) Prototype')
st.markdown(
    'Interactive prototype to compute the Standardised CVA capital charge (K) and RWA.\n\n'
    'This tool is for exploration and demonstrates the regulatory logic (CRR Art.384 SCVA). For regulatory reporting use a validated engine.'
)

# Sample data
SAMPLE = pd.DataFrame([
    {'TransactionID':'T1','CounterpartyID':'CPTY1','Instrument':'IRS','AssetClass':'rates','Notional':10000000.0,'RemainingMaturity':5.0,'EAD':1709678.88,'MTM':0.0,'Collateral':0.0,'SupervisoryRW':0.03},
    {'TransactionID':'T2','CounterpartyID':'CPTY1','Instrument':'IRS','AssetClass':'rates','Notional':2000000.0,'RemainingMaturity':2.0,'EAD':200000.0,'MTM':0.0,'Collateral':0.0,'SupervisoryRW':0.03},
    {'TransactionID':'T3','CounterpartyID':'CPTY2','Instrument':'FX Swap','AssetClass':'fx','Notional':5000000.0,'RemainingMaturity':3.0,'EAD':500000.0,'MTM':0.0,'Collateral':0.0,'SupervisoryRW':0.10}
])

# --- Sidebar: Configuration & Regulatory Panel ---
st.sidebar.header('Calculation options')
method = st.sidebar.selectbox('Method', ['SCVA (Art.384)', 'OEM - Original Exposure Method (Art.385)', 'ACVA (Internal model - placeholder)'])
apply_eu_exemptions = st.sidebar.checkbox('Apply EU exemptions (illustrative)', value=True)

ead_source = st.sidebar.radio('EAD source', ['User-provided EAD (column)', 'Simple SA-CCR estimator (demo)', 'Upload SA-CCR CSV'])
split_enabled = st.sidebar.checkbox('Enable 50/50 split scenario', value=True)
split_counterparty = st.sidebar.text_input('Counterparty to split 50/50 (enter ID)', value='CPTY1' if split_enabled else '')

st.sidebar.markdown('---')
st.sidebar.header('Regulatory panel (dynamic)')
# Show dynamic conclusions in the panel
st.sidebar.markdown(f"**Method used:** {method}")
if method.startswith('SCVA'):
    st.sidebar.markdown('- CRR reference: **Art.384** — Standardised CVA formula and supervisory weights')
elif method.startswith('OEM'):
    st.sidebar.markdown('- CRR reference: **Art.385** — Original Exposure Method (OEM) alternative)')
else:
    st.sidebar.markdown('- ACVA requires an approved internal model and supervisory permission (not computed in this prototype). See CRR Art.384(1) and supervisory guidance.')

st.sidebar.markdown(f"**EU exemptions applied:** {'Yes' if apply_eu_exemptions else 'No'}")
if apply_eu_exemptions:
    st.sidebar.markdown('- CRR ref: **Art.382 / 382(5)** (exemptions) and EBA Q&As. EU exemptions historically allowed some non-financial counterparties or short-term trades to be exempt — check EBA/ACPR notes in your dataset.')

st.sidebar.markdown(f"**EAD source:** {ead_source}")
if ead_source.startswith('Simple'):
    st.sidebar.markdown('- CRR ref for SA-CCR: **Arts.274–282**. This prototype uses a simplified estimator for quick exploration only.')
else:
    st.sidebar.markdown('- Use your SA-CCR engine or upload SA-CCR file with EADs per trade.')

# Hedging inputs
st.sidebar.markdown('---')
st.sidebar.header('Hedges / Eligible credit hedges')
hedges_recognised = st.sidebar.checkbox('Recognised eligible hedges (manual)')
hedge_adjustment_pct = st.sidebar.number_input('Manual adjustment to portfolio K (percent, e.g. -10 for 10% reduction)', value=0.0, step=0.1)
if hedges_recognised:
    st.sidebar.markdown('- Note: Eligible hedges recognition is complex (CRR + EBA). This checkbox is a manual indicator; you must validate hedges under rules and provide adjustments manually.')

st.sidebar.markdown('---')
st.sidebar.header('Useful references (uploaded files)')
st.sidebar.markdown('- regulation 573 2013 (CRR) CELEX_32013R0575_EN.pdf')
st.sidebar.markdown('- regulation 575-2013 modified version 2025 CRR CELEX_02013R0575-20240709_EN_TXT.pdf')
st.sidebar.markdown('- 2024 03 25_revue_acpr_paquet_bancaire.pdf')
st.sidebar.markdown('- 2024 12 30_Notice_CRD4_marques_revision.pdf')
st.sidebar.markdown('- EBA Report on CVA.pdf')

# --- Main area: Upload / edit transactions ---
col_main, col_right = st.columns([3,1])
with col_main:
    st.subheader('Transaction input (upload CSV or edit sample)')
    uploaded = st.file_uploader('Upload transactions CSV (optional). Required columns: TransactionID, CounterpartyID, Instrument, AssetClass, Notional, RemainingMaturity, EAD, MTM, SupervisoryRW', type=['csv'])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f'Failed to read CSV: {e}')
            df = SAMPLE.copy()
    else:
        df = SAMPLE.copy()

    # Allow in-place editing (experimental_data_editor in newer Streamlit versions)
    try:
        edited = st.experimental_data_editor(df, num_rows='dynamic', use_container_width=True)
    except Exception:
        edited = st.dataframe(df)
        st.info('Editable data editor not available; upload a CSV to change inputs.')

    # If EAD source is simple estimator, fill missing EADs
    if ead_source.startswith('Simple'):
        for i, row in edited.iterrows():
            if 'EAD' not in edited.columns or pd.isna(row.get('EAD', None)) or float(row.get('EAD', 0)) == 0:
                edited.at[i, 'EAD'] = simple_sa_ccr_estimate(row.get('Notional', 0), asset_class=row.get('AssetClass', 'rates'))

    # If user uploaded SA-CCR file, allow mapping
    if ead_source.startswith('Upload'):
        ead_file = st.file_uploader('Upload SA-CCR results CSV with TransactionID and EAD columns', type=['csv'], key='eadupload')
        if ead_file is not None:
            ead_df = pd.read_csv(ead_file)
            if 'TransactionID' in ead_df.columns and 'EAD' in ead_df.columns:
                edited = edited.merge(ead_df[['TransactionID','EAD']], on='TransactionID', how='left', suffixes=('','_uploaded'))
                # prefer uploaded EAD if present
                edited['EAD'] = edited['EAD_uploaded'].combine_first(edited['EAD']) if 'EAD_uploaded' in edited.columns else edited['EAD']
                edited = edited.drop(columns=[c for c in edited.columns if c.endswith('_uploaded')])
            else:
                st.error('SA-CCR CSV must contain TransactionID and EAD columns')

    st.markdown('---')
    st.subheader('Computation')
    # Compute DF and per-txn term
    edited['DF'] = edited['RemainingMaturity'].apply(lambda m: discount_factor(float(m) if not pd.isna(m) else 0.0))
    edited['Term'] = edited.apply(lambda r: single_counterparty_term(r['SupervisoryRW'], r['RemainingMaturity'], r['EAD'], r['DF']), axis=1)

    # Aggregate per counterparty
    agg = edited.groupby('CounterpartyID').agg(Notional=('Notional','sum'), EAD=('EAD','sum'), EffectiveMaturity=('RemainingMaturity','max'), AvgDF=('DF','mean'), SupervisoryRW=('SupervisoryRW','first'), TermSum=('Term','sum')).reset_index()
    agg['t_i'] = agg['TermSum']

    # Apply EU exemptions (illustrative heuristic)
    if apply_eu_exemptions:
        def is_exempt_row(r):
            # Heuristic: small notional + very low RW -> illustrate exemption
            return (r['SupervisoryRW'] <= 0.03) and (r['Notional'] <= 15_000_000)
        agg['is_exempt'] = agg.apply(is_exempt_row, axis=1)
        agg['t_i_used'] = agg.apply(lambda r: 0.0 if r['is_exempt'] else r['t_i'], axis=1)
    else:
        agg['is_exempt'] = False
        agg['t_i_used'] = agg['t_i']

    # If OEM selected, we leave t_i_used as-is here (OEM would use different Mi calculation in full implementation). Mark method.

    # Portfolio K
    terms = list(agg['t_i_used'].astype(float).fillna(0.0))
    K = portfolio_K(terms, corr=0.25, multiplier=2.33)
    RWA = K * 12.5

    # 50/50 split scenario
    split_results = None
    if split_enabled and split_counterparty:
        if split_counterparty in edited['CounterpartyID'].values:
            mask = edited['CounterpartyID'] == split_counterparty
            to_split = edited[mask].copy()
            a = to_split.copy(); a['CounterpartyID'] = a['CounterpartyID'] + '_A'; a['EAD'] = a['EAD']*0.5; a['Notional'] = a['Notional']*0.5
            b = to_split.copy(); b['CounterpartyID'] = b['CounterpartyID'] + '_B'; b['EAD'] = b['EAD']*0.5; b['Notional'] = b['Notional']*0.5
            others = edited[~mask].copy()
            split_full = pd.concat([others, a, b], ignore_index=True)
            split_full['DF'] = split_full['RemainingMaturity'].apply(lambda m: discount_factor(float(m) if not pd.isna(m) else 0.0))
            split_full['Term'] = split_full.apply(lambda r: single_counterparty_term(r['SupervisoryRW'], r['RemainingMaturity'], r['EAD'], r['DF']), axis=1)
            agg_s = split_full.groupby('CounterpartyID').agg(TermSum=('Term','sum')).reset_index()
            K_split = portfolio_K(list(agg_s['TermSum']), corr=0.25, multiplier=2.33)
            RWA_split = K_split * 12.5
            split_results = {'K_split':K_split, 'RWA_split':RWA_split}

    # Apply manual hedge adjustment to K if user indicates hedges
    if hedges_recognised and hedge_adjustment_pct != 0.0:
        K_adj = K * (1.0 + hedge_adjustment_pct / 100.0)
        RWA_adj = K_adj * 12.5
    else:
        K_adj = K
        RWA_adj = RWA

    st.subheader('Aggregated per-counterparty')
    st.dataframe(agg[['CounterpartyID','Notional','EAD','EffectiveMaturity','SupervisoryRW','t_i','is_exempt','t_i_used']])

    st.subheader('Portfolio results')
    st.metric('K (base)', f"{K:.2f}")
    st.metric('RWA (base)', f"{RWA:.2f}")
    if hedges_recognised and hedge_adjustment_pct != 0.0:
        st.metric('K (after hedge adj)', f"{K_adj:.2f}")
        st.metric('RWA (after hedge adj)', f"{RWA_adj:.2f}")

    if split_results is not None:
        st.subheader('50/50 split scenario')
        st.write(split_results)

    # Download results
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        edited.to_excel(writer, sheet_name='Inputs', index=False)
        agg.to_excel(writer, sheet_name='Counterparty_Aggregated', index=False)
        pd.DataFrame([{'Metric':'K','Value':K},{'Metric':'RWA','Value':RWA}]).to_excel(writer, sheet_name='Portfolio_Results', index=False)
        if split_results is not None:
            pd.DataFrame([split_results]).to_excel(writer, sheet_name='Split_Results', index=False)
    out.seek(0)
    st.download_button('Download results (Excel)', data=out, file_name='CVA_results.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

with col_right:
    st.subheader('Regulatory panel (summary)')
    st.markdown('**Active method**')
    st.write(method)
    st.markdown('**CRR Articles referenced**')
    st.write('- Art.381-386: CVA own funds (general)')
    st.write('- Art.384: Standardised CVA (formula & supervisory weights)')
    st.write('- Art.385: Original Exposure Method (alternative)')
    st.write('- Arts.274-282: SA-CCR (EAD source)')

    st.markdown('**Exemptions**')
    st.write('Applied' if apply_eu_exemptions else 'Not applied')
    st.markdown('**Hedges**')
    st.write('Recognised (manual)' if hedges_recognised else 'Not recognised')

    st.markdown('**Ambiguities & supervisory notes (short)**')
    st.markdown('- EU exemptions have historically created divergences with Basel; check EBA Q&As and ACPR notes.')
    st.markdown('- SA-CCR EAD should come from your production SA-CCR engine; this app uses a simplified estimator if selected.')
    st.markdown('- OEM vs SCVA: OEM is an alternative; SCVA is the standardised formula used for comparability.')
    st.markdown('- ACVA: requires internal model approval (not implemented here).')

    st.markdown('**Useful uploaded documents (check canvas files)**')
    st.write('- regulation 573 2013 (CRR) CELEX_32013R0575_EN.pdf')
    st.write('- 2024 03 25_revue_acpr_paquet_bancaire.pdf')
    st.write('- 2024 12 30_Notice_CRD4_marques_revision.pdf')
    st.write('- EBA Report on CVA.pdf')

# End of app
