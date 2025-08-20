import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt

st.set_page_config(page_title="SaaS Valuation & MRR Dashboard", layout="wide")

# ------------- Helpers -------------
@st.cache_data(show_spinner=False)
def load_excel(file) -> dict:
    xls = pd.ExcelFile(file)
    sheets = {name: pd.read_excel(xls, name) for name in xls.sheet_names}
    return sheets

def ensure_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas obligatorias: {missing} en la hoja.")
        st.stop()

def safe_div(a, b):
    try:
        return (a / b) if (b not in (None, 0) and not pd.isna(b)) else np.nan
    except Exception:
        return np.nan

def monthly_fifo_cohorts(df):
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"]).dt.to_period("M").dt.to_timestamp()
    work = work.sort_values(["Plan", "Date"]).reset_index(drop=True)
    results = []
    for plan, g in work.groupby("Plan", sort=False):
        queue = []
        initial_map = {}
        for month, gm in g.groupby("Date", sort=True):
            new_c = int(gm["New Customers"].sum())
            lost_c = int(gm["Lost Customers"].sum())
            if new_c > 0:
                queue.append([month, new_c])
                initial_map[month] = initial_map.get(month, 0) + new_c
            remaining_to_remove = lost_c
            qi = 0
            while remaining_to_remove > 0 and qi < len(queue):
                cohort_month, remaining = queue[qi]
                take = min(remaining, remaining_to_remove)
                queue[qi][1] -= take
                remaining_to_remove -= take
                if queue[qi][1] == 0:
                    qi += 1
            queue = [q for q in queue if q[1] > 0]
            for cohort_month, remaining in queue:
                results.append({
                    "Plan": plan,
                    "Cohort": cohort_month,
                    "Month": month,
                    "Remaining": remaining,
                    "Initial": initial_map.get(cohort_month, np.nan)
                })
    if not results:
        return pd.DataFrame(), pd.DataFrame()
    res = pd.DataFrame(results)
    res["Age (months)"] = ((res["Month"].dt.year - res["Cohort"].dt.year) * 12 +
                           (res["Month"].dt.month - res["Cohort"].dt.month)).astype(int)
    res = res[~res["Initial"].isna() & (res["Initial"] > 0)]
    res["Retention %"] = (res["Remaining"] / res["Initial"]) * 100
    total = (res.groupby(["Cohort", "Age (months)"])["Remaining"].sum().reset_index()
               .merge(res.groupby(["Cohort"])["Initial"].sum().reset_index(), on="Cohort", suffixes=("", "_cohort")))
    total["Retention %"] = (total["Remaining"] / total["Initial_cohort"]) * 100
    pivot_total = total.pivot(index="Cohort", columns="Age (months)", values="Retention %").sort_index()
    res["Weight"] = res["Initial"]
    res_plan = (res.groupby(["Plan", "Cohort", "Age (months)"])
                  .apply(lambda x: np.average(x["Retention %"], weights=x["Weight"]))
                  .reset_index(name="Retention %"))
    pivot_plan = res_plan.pivot_table(index=["Plan","Cohort"], columns="Age (months)", values="Retention %")
    return pivot_total, pivot_plan

def compute_components(df_data, df_prices):
    d = df_data.copy()
    d["Date"] = pd.to_datetime(d["Date"]).dt.to_period("M").dt.to_timestamp()
    p = df_prices.copy()
    d = d.merge(p[["Plan", "Price MRR (€)", "Multiple (x ARR)"]], on="Plan", how="left")
    d = d.sort_values(["Plan","Date"]).reset_index(drop=True)
    d["New MRR (€)"]       = d["New Customers"]  * d["Price MRR (€)"]
    d["Churned MRR (€)"]   = d["Lost Customers"] * d["Price MRR (€)"]
    if "Real MRR (optional €)" in d.columns and d["Real MRR (optional €)"].notna().any():
        d["MRR (€)"] = d["Real MRR (optional €)"]
    elif "Active Customers (optional)" in d.columns and d["Active Customers (optional)"].notna().any():
        d["MRR (€)"] = d["Active Customers (optional)"] * d["Price MRR (€)"]
    else:
        d["MRR (€)"] = d.groupby("Plan")["New MRR (€)"].cumsum() - d.groupby("Plan")["Churned MRR (€)"].cumsum()
    d["ΔMRR (€)"] = d.groupby("Plan")["MRR (€)"].diff().fillna(d["MRR (€)"])
    residual = d["ΔMRR (€)"] - d["New MRR (€)"] + d["Churned MRR (€)"]
    d["Expansion MRR (inferred €)"]  = residual.clip(lower=0)
    d["Downgraded MRR (inferred €)"] = (-residual).clip(lower=0)
    if "Active Customers (optional)" in d.columns and d["Active Customers (optional)"].notna().any():
        d["Active Customers"] = d["Active Customers (optional)"]
    else:
        d["Active Customers"] = d.groupby("Plan").apply(
            lambda g: g["New Customers"].cumsum() - g["Lost Customers"].cumsum()
        ).reset_index(level=0, drop=True)
    return d

def ytd_metrics(monthly, year):
    this_year = monthly[monthly["Date"].dt.year == year].copy()
    if this_year.empty:
        return {}
    this_year = this_year.sort_values("Date")
    start_mrr = this_year.iloc[0]["Start MRR (€)"]
    if pd.isna(start_mrr):
        start_mrr = this_year.iloc[0]["Total MRR (€)"]
    end_mrr = this_year.iloc[-1]["Total MRR (€)"]
    growth_ytd = safe_div(end_mrr - start_mrr, start_mrr) * 100 if start_mrr else np.nan
    churn_plus_contr = this_year["Churned MRR (€)"].sum() + this_year["Downgraded MRR (inferred €)"].sum()
    expansion = this_year["Expansion MRR (inferred €)"].sum()
    grr_ytd = (1 - safe_div(churn_plus_contr, start_mrr)) * 100 if start_mrr else np.nan
    nrr_ytd = (1 + safe_div(expansion - churn_plus_contr, start_mrr)) * 100 if start_mrr else np.nan
    net_new_ytd = this_year["Net New MRR (€)"].sum()
    return dict(growth_ytd=growth_ytd, grr_ytd=grr_ytd, nrr_ytd=nrr_ytd, net_new_ytd=net_new_ytd)

# ------------- UI -------------
st.title("📊 SaaS Valuation & MRR Dashboard")
st.caption("Sube tu Excel (hojas mínimas: **Prices** y **Data**). Opcional: **CAC**.")

uploaded = st.file_uploader("Cargar Excel (.xlsx)", type=["xlsx"])
if uploaded is None:
    st.warning("Sube un Excel con hojas **Prices** y **Data** (opcional **CAC**).")
    st.stop()

# Cargar hojas
sheets = load_excel(uploaded)

# Validaciones
ensure_columns(sheets["Prices"], ["Plan","Price MRR (€)","Price ARR (€)","Multiple (x ARR)"])
ensure_columns(sheets["Data"], ["Date","Plan","New Customers","Lost Customers","Active Customers (optional)","Real MRR (optional €)"])

# Dataframes base
df_prices = sheets["Prices"].copy()
df_data = sheets["Data"].copy()
df_cac  = sheets.get("CAC", pd.DataFrame(columns=["Date","Sales & Marketing Spend (€)","New Customers"])).copy()

# Normalizaciones
df_data["Date"] = pd.to_datetime(df_data["Date"]).dt.to_period("M").dt.to_timestamp()
df_prices["Plan"] = df_prices["Plan"].astype(str)

# Componentes MRR a nivel plan/mes
comp = compute_components(df_data, df_prices)

# Agregados a TOTAL (todas las tarifas)
monthly = (comp.groupby("Date", as_index=False)
            .agg({
                "New Customers":"sum",
                "Lost Customers":"sum",
                "Active Customers":"sum",
                "New MRR (€)":"sum",
                "Expansion MRR (inferred €)":"sum",
                "Churned MRR (€)":"sum",
                "Downgraded MRR (inferred €)":"sum",
                "MRR (€)":"sum"
            })
          ).sort_values("Date")
monthly = monthly.rename(columns={"MRR (€)":"Total MRR (€)"})
monthly["Start MRR (€)"] = monthly["Total MRR (€)"].shift(1)
monthly["Net New MRR (€)"] = (monthly["New MRR (€)"] + monthly["Expansion MRR (inferred €)"]
                              - monthly["Churned MRR (€)"] - monthly["Downgraded MRR (inferred €)"])
monthly["Total ARR (€)"] = monthly["Total MRR (€)"] * 12
monthly["GRR %"] = (1 - (monthly["Churned MRR (€)"] + monthly["Downgraded MRR (inferred €)"]) / monthly["Start MRR (€)"]) * 100
monthly["NRR %"] = (1 + (monthly["Expansion MRR (inferred €)"] - monthly["Churned MRR (€)"] - monthly["Downgraded MRR (inferred €)"]) / monthly["Start MRR (€)"]) * 100
monthly["GRR %"] = monthly["GRR %"].replace([np.inf, -np.inf], np.nan)
monthly["NRR %"] = monthly["NRR %"].replace([np.inf, -np.inf], np.nan)
monthly["MoM Growth %"] = ((monthly["Total MRR (€)"] - monthly["Start MRR (€)"]) / monthly["Start MRR (€)"] * 100).replace([np.inf, -np.inf], np.nan)
monthly["ARPU (€)"] = monthly["Total MRR (€)"] / monthly["Active Customers"].replace(0, np.nan)
monthly["Quick Ratio"] = (monthly["New MRR (€)"] + monthly["Expansion MRR (inferred €)"]) / (monthly["Churned MRR (€)"] + monthly["Downgraded MRR (inferred €)"]).replace(0, np.nan)

# --------- Filtros ---------
years = sorted(monthly["Date"].dt.year.unique())
default_year = years[-1] if years else datetime.now().year
col1, col2, col3 = st.columns([1,1,1])
with col1:
    sel_years = st.multiselect("Año(s)", options=years, default=[default_year] if years else [])
with col2:
    sector = st.selectbox("Sector/Perfil", [
        "Horizontal SaaS", "Vertical SaaS", "PLG", "Enterprise", "Fintech SaaS", "Health SaaS", "DevTools", "Otro"
    ])
with col3:
    gross_margin = st.slider("Margen bruto (%) para LTV", 40, 95, 80, step=1)

# Filtrado por años
filt = monthly[monthly["Date"].dt.year.isin(sel_years)].copy() if sel_years else monthly.copy()

# --------- KPIs (top) - safer rendering ---------
if filt.empty:
    st.error("No hay datos después de aplicar los filtros. Revisa el Excel o los años seleccionados.")
    st.stop()

last_row = filt.sort_values("Date").iloc[-1]
active_now = last_row.get("Active Customers", np.nan)
mrr_now = last_row.get("Total MRR (€)", np.nan)
arr_now = mrr_now * 12 if pd.notna(mrr_now) else np.nan

# CAC desde hoja CAC (en filtros por año)
cac_value = np.nan
if not df_cac.empty:
    df_cac["Date"] = pd.to_datetime(df_cac["Date"]).dt.to_period("M").dt.to_timestamp()
    cac_year = df_cac[df_cac["Date"].dt.year.isin(sel_years)] if sel_years else df_cac.copy()
    total_spend = cac_year["Sales & Marketing Spend (€)"].sum(min_count=1)
    if "New Customers" in cac_year.columns:
        total_new = cac_year["New Customers"].sum(min_count=1)
    elif "New Customers (from Data)" in cac_year.columns:
        total_new = cac_year["New Customers (from Data)"].sum(min_count=1)
    else:
        total_new = np.nan
    cac_value = safe_div(total_spend, total_new)

avg_active = filt["Active Customers"].replace(0,np.nan).mean()
churn_rate = safe_div(filt["Lost Customers"].sum(), avg_active)
arpu_now = safe_div(mrr_now, active_now)
ltv_value = safe_div(arpu_now * (gross_margin/100), churn_rate)
ltv_cac_ratio = safe_div(ltv_value, cac_value)

def fmt_money(x):
    return "—" if pd.isna(x) else ("€ {:,.0f}".format(float(x))).replace(",", ".")

def fmt_pct(x, decimals=1):
    return "—" if pd.isna(x) else (f"{float(x):.{decimals}f}%")

def fmt_plain(x, decimals=0):
    return "—" if pd.isna(x) else (f"{float(x):.{decimals}f}")

ytd = ytd_metrics(monthly, sel_years[-1] if sel_years else default_year)

k1, k2, k3, k4, k5, k6, k7, k8, k9 = st.columns(9)
k1.metric("Clientes activos", fmt_plain(active_now, 0))
k2.metric("MRR", fmt_money(mrr_now))
k3.metric("ARR", fmt_money(arr_now))
k4.metric("LTV/CAC", fmt_plain(ltv_cac_ratio, 2))
k5.metric("LTV (medio)", fmt_money(ltv_value))
k6.metric("Net New MRR (YTD)", fmt_money(ytd.get('net_new_ytd') if ytd else np.nan))
k7.metric("Growth YTD", fmt_pct(ytd.get('growth_ytd') if ytd else np.nan))
k8.metric("GRR YTD", fmt_pct(ytd.get('grr_ytd') if ytd else np.nan))
k9.metric("NRR YTD", fmt_pct(ytd.get('nrr_ytd') if ytd else np.nan))

st.divider()

# --------- Gráfico 1: Evolución de MRR ---------
st.subheader("Evolución de MRR")
line_mrr = alt.Chart(filt).mark_line(point=True).encode(
    x=alt.X('Date:T', title='Mes'),
    y=alt.Y('Total MRR (€):Q', title='MRR (€)'),
    tooltip=['Date:T','Total MRR (€):Q','New MRR (€):Q','Expansion MRR (inferred €):Q','Churned MRR (€):Q','Downgraded MRR (inferred €):Q']
).properties(height=300)
st.altair_chart(line_mrr, use_container_width=True)

# --------- Gráfico 2: Net New MRR (con filtros de componentes) ---------
st.subheader("NET NEW MRR por componentes")
components = {
    "New MRR (€)": st.checkbox("New", value=True),
    "Expansion MRR (inferred €)": st.checkbox("Expansion (inferred)", value=True),
    "Churned MRR (€)": st.checkbox("Churned", value=True),
    "Downgraded MRR (inferred €)": st.checkbox("Downgrade (inferred)", value=True)
}
stack_vars = [k for k,v in components.items() if v]
if stack_vars:
    stack_df = filt.melt(id_vars=["Date"], value_vars=stack_vars, var_name="Tipo", value_name="€")
    bars = alt.Chart(stack_df).mark_bar().encode(
        x=alt.X('Date:T', title='Mes'),
        y=alt.Y('sum(€):Q', title='€'),
        color=alt.Color('Tipo:N'),
        tooltip=['Date:T','Tipo:N','€:Q']
    ).properties(height=300)
    st.altair_chart(bars, use_container_width=True)
else:
    st.info("Activa al menos un componente para ver el gráfico de NET NEW MRR.")

st.divider()

# --------- Tabla por plan (filtros de año aplican) ---------
plans = comp[comp["Date"].dt.year.isin(sel_years)].copy() if sel_years else comp.copy()
last_by_plan = plans.sort_values("Date").groupby("Plan").tail(1)
tot_mrr = last_by_plan["MRR (€)"].sum() if not last_by_plan.empty else np.nan
table = (plans.groupby("Plan", as_index=False)
            .agg({
                "Active Customers":"last",
                "MRR (€)":"last",
                "New MRR (€)":"sum",
                "Expansion MRR (inferred €)":"sum",
                "Churned MRR (€)":"sum",
                "Downgraded MRR (inferred €)":"sum"
            }))
table["ARR (€)"] = table["MRR (€)"] * 12
table["Mix %"] = (table["MRR (€)"] / tot_mrr * 100).replace([np.inf,-np.inf], np.nan)
table = table.merge(df_prices[["Plan","Multiple (x ARR)"]], on="Plan", how="left")
table = table.rename(columns={
    "Active Customers":"Activos",
    "MRR (€)":"MRR (€)",
    "ARR (€)":"ARR (€)",
    "New MRR (€)":"New MRR (€)",
    "Expansion MRR (inferred €)":"Expansión MRR (€)",
    "Churned MRR (€)":"Churned MRR (€)",
    "Downgraded MRR (inferred €)":"Downgraded MRR (€)",
    "Multiple (x ARR)":"Múltiplo plan (x ARR)"
})
st.subheader("Desglose por plan")
st.dataframe(table.sort_values("MRR (€)", ascending=False), use_container_width=True)

st.divider()

# --------- Cohortes (total y por plan) ---------
st.subheader("Cohortes (aprox. FIFO) — total y por plan")
cohort_total, cohort_plan = monthly_fifo_cohorts(df_data[["Date","Plan","New Customers","Lost Customers"]])
cols = st.columns(2)
with cols[0]:
    st.markdown("**Cohortes — Total (retención %)**")
    if cohort_total.empty:
        st.info("No hay datos suficientes para cohortes.")
    else:
        st.dataframe(cohort_total.style.format('{:.0f}'), use_container_width=True, height=400)
with cols[1]:
    st.markdown("**Cohortes — Por plan (retención %)**")
    if cohort_plan.empty:
        st.info("No hay datos suficientes para cohortes por plan.")
    else:
        st.dataframe(cohort_plan.style.format('{:.0f}'), use_container_width=True, height=400)

st.divider()

# --------- Valoración ---------
st.subheader("Valoración — total y por plan")
base_multiples = {
    "Horizontal SaaS": 10, "Vertical SaaS": 9, "PLG": 12, "Enterprise": 8,
    "Fintech SaaS": 12, "Health SaaS": 9, "DevTools": 10, "Otro": 8
}
base_mult = base_multiples.get(sector, 10)
mult_slider = st.slider("Múltiplo xARR (ajustable según sector)", 4.0, 25.0, float(base_mult), 0.5)

valuation_total = arr_now * mult_slider if pd.notna(arr_now) else np.nan
st.metric("Valoración total (sector)", fmt_money(valuation_total))

per_plan_val = table[["Plan","ARR (€)","Múltiplo plan (x ARR)"]].copy()
per_plan_val["Valoración (€)"] = per_plan_val["ARR (€)"] * per_plan_val["Múltiplo plan (x ARR)"]
st.dataframe(per_plan_val.sort_values("Valoración (€)", ascending=False), use_container_width=True)

st.subheader("Simulador de valoración (3–5 años)")
c1, c2, c3 = st.columns(3)
with c1:
    sim_arr0 = st.number_input("ARR actual (€)", value=float(arr_now) if pd.notna(arr_now) else 0.0, step=1000.0, format="%.2f")
with c2:
    sim_growth_m = st.slider("Crecimiento mensual (%)", 0.0, 30.0, 5.0, 0.1)
with c3:
    sim_churn_m = st.slider("Churn mensual (%)", 0.0, 15.0, 2.0, 0.1)

net_g = (sim_growth_m - sim_churn_m) / 100.0
years = [3,4,5]
proj = []
for Y in years:
    arrY = sim_arr0 * ((1 + net_g) ** (12 * Y))
    growth_a = ( (1+sim_growth_m/100.0)**12 - 1 )
    churn_a  = ( (1+sim_churn_m/100.0)**12 - 1 )
    mult_adj = mult_slider * (1 + growth_a/0.40) * (1 - churn_a/0.20)
    mult_adj = max(mult_slider*0.5, min(mult_adj, mult_slider*2.0))
    valuationY = arrY * mult_slider
    valuationY_adj = arrY * mult_adj
    proj.append((f"{Y} años", arrY, valuationY, valuationY_adj, mult_adj))

df_proj = pd.DataFrame(proj, columns=["Horizonte","ARR proyectado (€)","Valor (múltiplo fijo)","Valor (ajustado crecimiento/churn)","Múltiplo ajustado"])
st.dataframe(df_proj.style.format({
    "ARR proyectado (€)": "€ {:,.0f}".format,
    "Valor (múltiplo fijo)": "€ {:,.0f}".format,
    "Valor (ajustado crecimiento/churn)": "€ {:,.0f}".format,
    "Múltiplo ajustado": "{:.2f}".format
}), use_container_width=True)

st.success("Listo. Sube un Excel actualizado cuando quieras para recalcular todo.")
