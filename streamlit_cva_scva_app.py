"""
Streamlit prototype: CVA SCVA calculator with GitHub PDF preview + article text extraction

Changes:
- PDF preview now uses GitHub raw link instead of embedding full Base64 (avoids white box issue).
- If GitHub URL fails, falls back to showing extracted text with pdfplumber.
- Extracted article text is highlighted (article heading bold + blue).
- Sidebar keeps external links (EUR-Lex + Custom GPT).
- Excel/CSV export logic unchanged.
- Improved article detection with regex (handles 'Art. 386', 'ARTICLE 386', etc.).
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import io
import re
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
    return add_on

# PDF utils
PDF_FILENAME = "CRR_575_2013.pdf"
GITHUB_PDF_URL = "https://raw.githubusercontent.com/atleshova/cva-calculator/main/CRR_575_2013.pdf"

def find_article_pages_and_text(pdf_path, articles):
    mapping, texts = {}, {}
    if not PDFPLUMBER_AVAILABLE:
        return mapping, texts
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "")
                lower = text.lower()
                for art in articles:
                    num = art.split()[1]  # e.g. "386"
                    # Match 'Article 386', 'Art. 386', '386.' etc.
                    if re.search(rf"(article|art\.?)[\s\u00A0]*{num}\b", lower) and art not in mapping:
                        mapping[art] = i
                        texts[art] = text
        return mapping, texts
    except Exception:
        return {}, {}

# --- Sample data ---
SAMPLE = pd.DataFrame([
    {'TransactionID':'T1','CounterpartyID':'CPTY1','Instrument':'IRS','AssetClass':'rates','Notional':1e7,'RemainingMaturity':5.0,'EAD':1709678.88,'MTM':0.0,'Collateral':0.0,'SupervisoryRW':0.03},
    {'TransactionID':'T2','CounterpartyID':'CPTY1','Instrument':'IRS','AssetClass':'rates','Notional':2e6,'RemainingMaturity':2.0,'EAD':200000.0,'MTM':0.0,'Collateral':0.0,'SupervisoryRW':0.03},
    {'TransactionID':'T3','CounterpartyID':'CPTY2','Instrument':'FX Swap','AssetClass':'fx','Notional':5e6,'RemainingMaturity':3.0,'EAD':500000.0,'MTM':0.0,'Collateral':0.0,'SupervisoryRW':0.10}
])

# --- Sidebar ---
st.sidebar.header("Calculation options")
method = st.sidebar.selectbox("Method", ["SCVA (Art.384)", "OEM (Art.385)", "ACVA (Internal model - placeholder)"])
apply_eu_exemptions = st.sidebar.checkbox("Apply EU exemptions (illustrative)", value=True)
ead_source = st.sidebar.radio("EAD source", ["User-provided", "Simple SA-CCR estimator", "Upload SA-CCR CSV"])
split_enabled = st.sidebar.checkbox("Enable 50/50 split", value=True)
split_counterparty = st.sidebar.text_input("Counterparty to split 50/50", value="CPTY1" if split_enabled else "")

st.sidebar.markdown("---")
st.sidebar.header("Regulatory panel (click an article to preview)")
ARTICLES = [f"Article {i}" for i in range(381, 387)]
sel_article = st.sidebar.radio("Open article", ARTICLES)

st.sidebar.markdown(f"**Method used:** {method}")
st.sidebar.markdown(f"**EU exemptions applied:** {'Yes' if apply_eu_exemptions else 'No'}")
st.sidebar.markdown(f"**EAD source:** {ead_source}")

st.sidebar.markdown("---")
st.sidebar.markdown("**External tools / links**")
st.sidebar.markdown("- [EUR-Lex: CRR Regulation](https://eur-lex.europa.eu/eli/reg/2013/575/oj/eng)")
CUSTOM_GPT_URL = "https://chat.openai.com/g/g-68d3fd8f6cb881919a46c5e96e188006-cva-calculation"
st.sidebar.markdown(f"- [Custom GPT: CVA Calculation]({CUSTOM_GPT_URL})")

# --- Main ---
st.title("CVA — Standardised CVA (SCVA) Prototype")

uploaded = st.file_uploader("Upload transactions CSV (optional)", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        df = SAMPLE.copy()
else:
    df = SAMPLE.copy()

try:
    edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)
except Exception:
    edited = df.copy()
edited_df = pd.DataFrame(edited)

# Convert numerics
for col in ["RemainingMaturity", "EAD", "SupervisoryRW", "Notional"]:
    if col in edited_df.columns:
        edited_df[col] = pd.to_numeric(edited_df[col], errors="coerce").fillna(0.0)

# Fill EAD if needed
if ead_source.startswith("Simple"):
    for i, row in edited_df.iterrows():
        if row["EAD"] == 0:
            edited_df.at[i,"EAD"] = simple_sa_ccr_estimate(row["Notional"], asset_class=row["AssetClass"])

# DF + Terms
edited_df["DF"] = edited_df["RemainingMaturity"].apply(lambda m: discount_factor(float(m)))
edited_df["Term"] = edited_df.apply(lambda r: single_counterparty_term(r["SupervisoryRW"], r["RemainingMaturity"], r["EAD"], r["DF"]), axis=1)

# Aggregate
agg = edited_df.groupby("CounterpartyID").agg(
    Notional=("Notional","sum"),
    EAD=("EAD","sum"),
    EffectiveMaturity=("RemainingMaturity","max"),
    AvgDF=("DF","mean"),
    SupervisoryRW=("SupervisoryRW","first"),
    TermSum=("Term","sum")
).reset_index()
agg["t_i"] = agg["TermSum"]

if apply_eu_exemptions:
    agg["is_exempt"] = agg.apply(lambda r: (r["SupervisoryRW"]<=0.03 and r["Notional"]<=15_000_000), axis=1)
    agg["t_i_used"] = agg.apply(lambda r: 0.0 if r["is_exempt"] else r["t_i"], axis=1)
else:
    agg["is_exempt"] = False
    agg["t_i_used"] = agg["t_i"]

K = portfolio_K(list(agg["t_i_used"]))
RWA = 12.5*K

# --- Display ---
col_left, col_right = st.columns([3,2])
with col_left:
    st.subheader("Aggregated by counterparty")
    st.dataframe(agg)
    st.subheader("Portfolio results")
    st.metric("K", f"{K:.2f}")
    st.metric("RWA", f"{RWA:.2f}")

    # Export
    out = io.BytesIO()
    try:
        import openpyxl
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            edited_df.to_excel(writer, sheet_name="Inputs", index=False)
            agg.to_excel(writer, sheet_name="Counterparty_Aggregated", index=False)
            pd.DataFrame([{"Metric":"K","Value":K},{"Metric":"RWA","Value":RWA}]).to_excel(writer, sheet_name="Portfolio_Results", index=False)
        out.seek(0)
        st.download_button("Download results (Excel)", data=out, file_name="CVA_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        st.warning("Excel export unavailable — offering CSVs instead.")
        st.download_button("Download Inputs (CSV)", data=edited_df.to_csv(index=False).encode(), file_name="inputs.csv", mime="text/csv")
        st.download_button("Download Aggregated (CSV)", data=agg.to_csv(index=False).encode(), file_name="aggregated.csv", mime="text/csv")

with col_right:
    st.subheader("Regulation preview")

    # Always show the full PDF (Google Docs viewer, no page jump)
    PDF_VIEWER = f"https://docs.google.com/gview?embedded=true&url={GITHUB_PDF_URL}"
    st.components.v1.iframe(PDF_VIEWER, height=600)

    # Extracted text based on selection
    pdf_path = Path(PDF_FILENAME)
    if pdf_path.exists():
        mapping, texts = find_article_pages_and_text(str(pdf_path), ARTICLES)

        if sel_article in texts:
            st.subheader(f"Extracted text for {sel_article}")
            # Highlight article heading in blue
            highlighted = texts[sel_article].replace(sel_article, f"**:blue[{sel_article}]**")
            st.markdown(f"```text\n{highlighted}\n```")
        else:
            st.info(f"No extracted text available for {sel_article}.")
    else:
        st.info(f"Upload {PDF_FILENAME} in the repo root to enable preview.")
