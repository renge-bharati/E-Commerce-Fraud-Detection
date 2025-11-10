# Files for GitHub — E-commerce Fraud Detection App

This document contains three files you can copy into your GitHub repo:

1. `app.py` — Streamlit app (single-file) for exploring the Kaggle dataset.
2. `requirements.txt` — Python package requirements.
3. `README.md` — Repo README with instructions.

---

## app.py

```python
# app.py
# Streamlit app for exploring an e-commerce fraud dataset
# Put your CSV in the same folder and name it `ecommerce_fraud.csv` OR use the file uploader in the app.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(page_title="E‑commerce Fraud Explorer", layout="wide")

st.markdown("""
<style>
/* Tiny visual polish */
.block-container { padding-top: 1.25rem; }
.kpi { padding: 12px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
</style>
""", unsafe_allow_html=True)

st.title("🛒 E‑commerce Fraud Detection — Interactive Explorer")
st.write("Upload the dataset or place `ecommerce_fraud.csv` next to this app.")

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

# try to auto-load common filename, otherwise show uploader
df = None
try:
    df = load_csv("ecommerce_fraud.csv")
except Exception:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)

if df is None:
    st.warning("No dataset loaded. Please upload a CSV via the sidebar or place `ecommerce_fraud.csv` in the app folder.")
    st.stop()

# Basic cleaning: drop completely empty columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

st.sidebar.header("Data preview & filters")
st.sidebar.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

# Detect a 'fraud' label column automatically
possible_labels = ['is_fraud','isFraud','fraud','label','Class','is_fraud_label']
label_col = None
for c in df.columns:
    if c in possible_labels:
        label_col = c
        break
# fallback: look for binary column with only 0/1 or True/False
if label_col is None:
    for c in df.select_dtypes(include=[np.number, 'bool']).columns:
        vals = df[c].dropna().unique()
        if set(vals).issubset({0,1}) or set(vals).issubset({0.0,1.0}):
            label_col = c
            break

# Sidebar filters for some common columns
filters = {}
if 'customer_id' in df.columns:
    customers = df['customer_id'].astype(str).unique()
    sel = st.sidebar.multiselect('Customer', options=customers, default=None)
    if sel:
        filters['customer_id'] = sel

if 'merchant' in df.columns:
    merchants = df['merchant'].astype(str).unique()
    selm = st.sidebar.multiselect('Merchant', options=merchants, default=None)
    if selm:
        filters['merchant'] = selm

if 'amount' in df.columns:
    min_amt, max_amt = float(df['amount'].min()), float(df['amount'].max())
    r = st.sidebar.slider('Amount range', min_value=min_amt, max_value=max_amt, value=(min_amt, max_amt))
    filters['amount_range'] = r

# Apply filters
df_filtered = df.copy()
if 'customer_id' in filters:
    df_filtered = df_filtered[df_filtered['customer_id'].astype(str).isin(filters['customer_id'])]
if 'merchant' in filters:
    df_filtered = df_filtered[df_filtered['merchant'].astype(str).isin(filters['merchant'])]
if 'amount_range' in filters:
    lo, hi = filters['amount_range']
    df_filtered = df_filtered[df_filtered['amount'].between(lo, hi)]

# KPIs
st.markdown("#### Overview")
col1, col2, col3, col4 = st.columns([1.2,1.2,1.2,1.2])
with col1:
    st.markdown("<div class='kpi'>\n**Total transactions**<br/>\n**{:,}**\n</div>".format(len(df_filtered)), unsafe_allow_html=True)
with col2:
    if label_col is not None:
        fraud_count = int(df_filtered[label_col].astype(int).sum())
        st.markdown("<div class='kpi'>\n**Fraudulent**<br/>\n**{:,}**\n</div>".format(fraud_count), unsafe_allow_html=True)
    else:
        st.markdown("<div class='kpi'>\n**Fraudulent**<br/>\n**N/A**\n</div>", unsafe_allow_html=True)
with col3:
    if label_col is not None:
        rate = (df_filtered[label_col].astype(int).mean()) * 100
        st.markdown("<div class='kpi'>\n**Fraud rate**<br/>\n**{:.2f}%**\n</div>".format(rate), unsafe_allow_html=True)
    else:
        st.markdown("<div class='kpi'>\n**Fraud rate**<br/>\n**N/A**\n</div>", unsafe_allow_html=True)
with col4:
    if 'amount' in df.columns:
        st.markdown("<div class='kpi'>\n**Avg. amount**<br/>\n**{:.2f}**\n</div>".format(df_filtered['amount'].mean()), unsafe_allow_html=True)
    else:
        st.markdown("<div class='kpi'>\n**Avg. amount**<br/>\n**N/A**\n</div>", unsafe_allow_html=True)

st.markdown("---")

# Charts area
st.markdown("### Visualizations")

# 1) Fraud distribution
if label_col is not None:
    fig = px.pie(df_filtered, names=label_col, title='Fraud vs Non‑Fraud (Filtered)')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info('No fraud/label column detected automatically. If your dataset has a label column, rename it to one of: ' + ', '.join(possible_labels))

# 2) Amount distribution
if 'amount' in df.columns:
    fig2 = px.histogram(df_filtered, x='amount', nbins=50, title='Transaction amount distribution')
    st.plotly_chart(fig2, use_container_width=True)

# 3) Top merchants/customers by transaction count
cols_for_group = None
if 'merchant' in df.columns:
    cols_for_group = 'merchant'
elif 'customer_id' in df.columns:
    cols_for_group = 'customer_id'

if cols_for_group is not None:
    topn = 10
    grp = df_filtered[cols_for_group].astype(str).value_counts().nlargest(topn).reset_index()
    grp.columns = [cols_for_group, 'count']
    fig3 = px.bar(grp, x=cols_for_group, y='count', title=f'Top {topn} by transactions')
    st.plotly_chart(fig3, use_container_width=True)

# 4) Correlation heatmap for numeric columns
num_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) >= 2:
    corr = df_filtered[num_cols].corr()
    fig4, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig4)

st.markdown("---")

# Data table and download
st.markdown("### Sample rows (filtered)")
st.dataframe(df_filtered.head(200))

# download filtered
@st.cache_data
def convert_df_to_csv(df_in):
    return df_in.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df_filtered)
st.download_button(label='📥 Download filtered CSV', data=csv, file_name='filtered_ecommerce.csv', mime='text/csv')

st.markdown("---")

st.caption('Tip: add checkboxes and more detailed feature engineering when building models. This viewer is meant for exploration and quick reporting.')
```

---

## requirements.txt

```
streamlit>=1.25
pandas>=1.5
numpy>=1.23
plotly>=5.10
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.1
```

> You can pin versions if you prefer. The versions above are minimum recommended versions.

---

## README.md

````markdown
# E-commerce Fraud Explorer (Streamlit)

A simple Streamlit app to explore an e-commerce fraud detection dataset. This repository contains a single-file Streamlit app (`app.py`) plus `requirements.txt`.

## Files

- `app.py` — Streamlit app.
- `requirements.txt` — Python dependencies.
- (your dataset) `ecommerce_fraud.csv` — place this CSV file in the repo root or upload via the app UI.

## How to run locally

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate    # Windows
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your dataset in the repo root and name it `ecommerce_fraud.csv`, OR start the app and upload the CSV in the sidebar:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Deploy to Streamlit Community Cloud

1. Create a new GitHub repository and push this folder.
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) and connect your GitHub account.
3. Select the repo and the branch, set the command to `streamlit run app.py` and deploy.

## Notes & tips

* If the dataset's fraud label column has a different name, either rename it to one of: `is_fraud`, `isFraud`, `fraud`, `label`, `Class` or update `app.py` to point at your label column.
* Extend the app with model inference (scikit‑learn / joblib) or additional interactive charts.

## Source

Dataset: [https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset](https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset)

```

---

### Next steps

- Copy these three blocks into files in your repo: `app.py`, `requirements.txt`, `README.md`.
- If you want, I can also: create a `Procfile` for Heroku, add GitHub Actions for CI, or produce a cleaned/feature-engineered sample notebook for modeling. Tell me which you want next.

```
