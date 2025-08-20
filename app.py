import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt

st.set_page_config(page_title="SaaS Valuation & MRR Dashboard", layout="wide")

# =========================
# Helpers generales
# =========================
@st.cache_data(show_spinner=False)
def load_excel(file) -> dict:
    xls = pd.ExcelFile(file)
    sheets = {name: pd.read_excel(xls, name) for name in xls.sheet_names}
    return sheets

def ensure_columns(df, cols, sheet_name="(sheet)"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas obligatorias en **{sheet_name}**: {missing}")
        st.stop()

def safe_div(a, b):
    return (a / b) if (b is not None and b != 0 and not pd.isna(b)) else np.nan

def fmt_money(x):
    return f"€ {x:,.0f}".replace(",", ".") if pd.notna(x) else "—"

def fmt_pct(x, dec=1):
    return f"{x:.{dec}f}%" if pd.notna(x) else "—"

def fmt_plain(x, dec=0):
    if pd.isna(x):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{x:d}"
    try:
        return f"{x:.{dec}f}"
    except Exception:
        return str(x)

# =========================
# Cohortes FIFO (core)
# =========================
def monthly_fifo_cohorts(df):
    """
    Construye una matriz de cohorts aproximada (FIFO) con datos agregados mensuales.
    Requiere columnas: Date, Plan, New Customers, Lost Customers.
    Devuelve:
      - pivot_total: index=Cohort (mes alta), columns=Age (months), valores=Retention %
      - pivot_plan: index=(Plan, Cohort), columns=Age (months), valores=Retention %
    """
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"]).dt.to_period("M").dt.to_timestamp()
    work = work.sort_values(["Plan", "Date"]).reset_index(drop=True)

    results = []  # filas: plan, cohort_month, month, remaining, initial

    for plan, g in work.groupby("Plan", sort=False):
        queue = []  # lista de [cohort_month, remaining]
        initial_map = {}

        for month, gm in g.groupby("Date", sort=True):
            new_c = int(gm["New Customers"].sum())
            lost_c = int(gm["Lost Customers"].sum())

            if new_c > 0:
                queue.append([month, new_c])
                initial_map.setdefault(month, 0)
                initial_map[month] += new_c

            # Aplicamos bajas FIFO
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

            # Registrar estado de cada cohorte en este mes
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
    # Edad en meses
    res["Age (months)"] = (
        (res["Month"].dt.year - res["Cohort"].dt.year) * 12 +
        (res["Month"].dt.month - res["Cohort"].dt.month)
    ).astype(int)

    # Filtrar registros válidos
    res = res[~res["Initial"].isna() & (res["Initial"] > 0)]

    # Pivot TOTAL (suma global por cohorte → m0 = 100% correcto)
    total_rem = (res.groupby(["Cohort", "Age (months)"])["Remaining"]
                   .sum().reset_index())
    total_init = res.groupby("Cohort")["Initial"].sum()
    total_rem["Initial_total"] = total_rem["Cohort"].map(total_init)
    total_rem = total_rem[total_rem["Initial_total"] > 0]
    total_rem["Retention %"] = (total_rem["Remaining"] / total_rem["Initial_total"]) * 100.0
    pivot_total = total_rem.pivot(index="Cohort",
                                  columns="Age (months)",
                                  values="Retention %").sort_index()

    # Pivot POR PLAN (media ponderada por tamaño cohorte)
    res["Weight"] = res["Initial"]
    def _wavg(x):
        w = x["Weight"]
        return np.average((x["Remaining"] / x["Initial"]) * 100.0, weights=w)
    res_plan = (res.groupby(["Plan", "Cohort", "Age (months)"])
                  .apply(_wavg)
                  .reset_index(name="Retention %"))
    pivot_plan = res_plan.pivot_table(index=["Plan", "Cohort"],
                                      columns="Age (months)",
                                      values="Retention %")
    return pivot_total, pivot_plan

# =========================
# Carga inicial (upload only)
# =========================
st.title("📊 SaaS Valuation & MRR Dashboard")
st.caption("Sube tu Excel (hojas mínimas: **Prices** y **Data**). Opcional: **CAC**.")

uploaded = st.file_uploader("Cargar Excel (.xlsx)", type=["xlsx"])
if uploaded is None:
    st.info("Por favor, sube tu fichero Excel para continuar. (Requeridas: **Prices** y **Data**; opcional: **CAC**).")
    st.stop()

sheets = load_excel(uploaded)

# Validaciones
ensure_columns(sheets["Prices"], ["Plan","Price MRR (€)","Price ARR (€)","Multiple (x ARR)"], "Prices")
ensure_columns(sheets["Data"], ["Date","Plan","New Customers","Lost Customers","Active Customers (optional)","Real MRR (optional €)"], "Data")

# Dataframes base
df_prices = sheets["Prices"].copy()
df_data = sheets["Data"].copy()
df_cac  = sheets.get("CAC", pd.DataFrame(columns=["Date","Sales & Marketing Spend (€)","New Customers"])).copy()

# Normalizaciones
df_data["Date"] = pd.to_datetime(df_data["Date"]).dt.to_period("M").dt.to_timestamp()
df_prices["Plan"] = df_prices["Plan"].astype(str)

# =========================
# Cálculos principales
# =========================
def compute_components(df_data, df_prices):
    d = df_data.copy()
    d["Date"] = pd.to_datetime(d["Date"]).dt.to_period("M").dt.to_timestamp()
    p = df_prices.copy()
    d = d.merge(p[["Plan", "Price MRR (€)", "Multiple (x ARR)"]], on="Plan", how="left")

    d = d.sort_values(["Plan","Date"]).reset_index(drop=True)

    # Valores base
    d["New MRR (€)"]       = d["New Customers"]  * d["Price MRR (€)"]
    d["Churned MRR (€)"]   = d["Lost Customers"] * d["Price MRR (€)"]

    # MRR real por plan/mes
    if "Real MRR (optional €)" in d.columns and d["Real MRR (optional €)"].notna().any():
        d["MRR (€)"] = d["Real MRR (optional €)"]
    else:
        if "Active Customers (optional)" in d.columns and d["Active Customers (optional)"].notna().any():
            d["MRR (€)"] = d["Active Customers (optional)"] * d["Price MRR (€)"]
        else:
            d["MRR (€)"] = d.groupby("Plan")["New MRR (€)"].cumsum() - d.groupby("Plan")["Churned MRR (€)"].cumsum()

    # Diferencia mensual por plan
    d["ΔMRR (€)"] = d.groupby("Plan")["MRR (€)"].diff().fillna(d["MRR (€)"])

    # Residuo para inferir expansión/contracción
    residual = d["ΔMRR (€)"] - d["New MRR (€)"] + d["Churned MRR (€)"]
    d["Expansion MRR (inferred €)"]  = residual.clip(lower=0)
    d["Downgraded MRR (inferred €)"] = (-residual).clip(lower=0)

    # Clientes activos aprox si no vienen en fichero
    if "Active Customers (optional)" in d.columns and d["Active Customers (optional)"].notna().any():
        d["Active Customers"] = d["Active Customers (optional)"]
    else:
        d["Active Customers"] = d.groupby("Plan").apply(
            lambda g: g["New Customers"].cumsum() - g["Lost Customers"].cumsum()
        ).reset_index(level=0, drop=True)

    return d

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
          )
monthly = monthly.sort_values("Date")
monthly = monthly.rename(columns={"MRR (€)":"Total MRR (€)"})
monthly["Start MRR (€)"] = monthly["Total MRR (€)"].shift(1)
monthly["Net New MRR (€)"] = (monthly["New MRR (€)"] + monthly["Expansion MRR (inferred €)"]
                              - monthly["Churned MRR (€)"] - monthly["Downgraded MRR (inferred €)"])
monthly["Total ARR (€)"] = monthly["Total MRR (€)"] * 12
monthly["GRR %"] = (1 - (monthly["Churned MRR (€)"] + monthly["Downgraded MRR (inferred €)"])
                      / monthly["Start MRR (€)"]).replace([np.inf, -np.inf], np.nan) * 100
monthly["NRR %"] = (1 + (monthly["Expansion MRR (inferred €)"] - monthly["Churned MRR (€)"] - monthly["Downgraded MRR (inferred €)"])
                      / monthly["Start MRR (€)"]).replace([np.inf, -np.inf], np.nan) * 100
monthly["MoM Growth %"] = ((monthly["Total MRR (€)"] - monthly["Start MRR (€)"]) / monthly["Start MRR (€)"] * 100).replace([np.inf, -np.inf], np.nan)
monthly["ARPU (€)"] = monthly["Total MRR (€)"] / monthly["Active Customers"].replace(0, np.nan)
monthly["Quick Ratio"] = (monthly["New MRR (€)"] + monthly["Expansion MRR (inferred €)"]) / (monthly["Churned MRR (€)"] + monthly["Downgraded MRR (inferred €)"]).replace(0, np.nan)

# =========================
# Filtros
# =========================
years = sorted(monthly["Date"].dt.year.unique())
if not years:
    st.error("No hay fechas válidas en la hoja Data.")
    st.stop()

default_year = years[-1]
col1, col2, col3 = st.columns([1,1,1])
with col1:
    sel_years = st.multiselect("Año(s)", options=years, default=[default_year])
with col2:
    sector = st.selectbox("Sector/Perfil", [
        "Horizontal SaaS", "Vertical SaaS", "PLG", "Enterprise", "Fintech SaaS", "Health SaaS", "DevTools", "Otro"
    ])
with col3:
    gross_margin = st.slider("Margen bruto (%) para LTV", 40, 95, 80, step=1)

base_multiples = {
    "Horizontal SaaS": 10, "Vertical SaaS": 9, "PLG": 12, "Enterprise": 8,
    "Fintech SaaS": 12, "Health SaaS": 9, "DevTools": 10, "Otro": 8
}
base_mult = base_multiples.get(sector, 10)

filt = monthly[monthly["Date"].dt.year.isin(sel_years)].copy()
if filt.empty:
    st.warning("No hay datos para el año seleccionado. Ajusta el filtro de año.")
    st.stop()

# =========================
# KPIs
# =========================
last_row = filt.sort_values("Date").iloc[-1]
active_now = int(last_row["Active Customers"]) if "Active Customers" in last_row else np.nan
mrr_now = float(last_row["Total MRR (€)"]) if "Total MRR (€)" in last_row else np.nan
arr_now = mrr_now * 12 if pd.notna(mrr_now) else np.nan

# CAC desde hoja CAC
cac_value = np.nan
if not df_cac.empty:
    df_cac["Date"] = pd.to_datetime(df_cac["Date"]).dt.to_period("M").dt.to_timestamp()
    cac_year = df_cac[df_cac["Date"].dt.year.isin(sel_years)]
    total_spend = cac_year["Sales & Marketing Spend (€)"].sum(min_count=1)
    if "New Customers" in cac_year.columns:
        total_new = cac_year["New Customers"].sum(min_count=1)
    elif "New Customers (from Data)" in cac_year.columns:
        total_new = cac_year["New Customers (from Data)"].sum(min_count=1)
    else:
        total_new = np.nan
    cac_value = safe_div(total_spend, total_new)

avg_active = filt["Active Customers"].replace(0, np.nan).mean()
churn_rate = safe_div(filt["Lost Customers"].sum(), avg_active)  # mensual aprox
arpu_now = safe_div(mrr_now, active_now)
ltv_value = safe_div(arpu_now * (gross_margin/100), churn_rate)
ltv_cac_ratio = safe_div(ltv_value, cac_value)

def ytd_metrics(monthly, year):
    this_year = monthly[monthly["Date"].dt.year == year].copy()
    if this_year.empty:
        return {}
    this_year = this_year.sort_values("Date")
    end = this_year.iloc[-1]
    start_mrr = this_year.iloc[0]["Start MRR (€)"] if not pd.isna(this_year.iloc[0]["Start MRR (€)"]) else this_year.iloc[0]["Total MRR (€)"]
    growth_ytd = safe_div(end["Total MRR (€)"] - start_mrr, start_mrr) * 100 if start_mrr else np.nan
    churn_plus_contr = this_year["Churned MRR (€)"].sum() + this_year["Downgraded MRR (inferred €)"].sum()
    expansion = this_year["Expansion MRR (inferred €)"].sum()
    grr_ytd = (1 - safe_div(churn_plus_contr, start_mrr)) * 100 if start_mrr else np.nan
    nrr_ytd = (1 + safe_div(expansion - churn_plus_contr, start_mrr)) * 100 if start_mrr else np.nan
    net_new_ytd = this_year["Net New MRR (€)"].sum()
    return dict(growth_ytd=growth_ytd, grr_ytd=grr_ytd, nrr_ytd=nrr_ytd, net_new_ytd=net_new_ytd)

ytd = ytd_metrics(monthly, sel_years[-1] if sel_years else default_year)

k1, k2, k3, k4, k5, k6, k7, k8, k9 = st.columns(9)
k1.metric("Clientes activos", fmt_plain(active_now))
k2.metric("MRR", fmt_money(mrr_now))
k3.metric("ARR", fmt_money(arr_now))
k4.metric("LTV/CAC", fmt_plain(ltv_cac_ratio, dec=2))
k5.metric("LTV (medio)", fmt_money(ltv_value))
k6.metric("Net New MRR (YTD)", fmt_money(ytd.get("net_new_ytd", np.nan) if ytd else np.nan))
k7.metric("Growth YTD", fmt_pct(ytd.get("growth_ytd", np.nan) if ytd else np.nan))
k8.metric("GRR YTD", fmt_pct(ytd.get("grr_ytd", np.nan) if ytd else np.nan))
k9.metric("NRR YTD", fmt_pct(ytd.get("nrr_ytd", np.nan) if ytd else np.nan))

st.divider()

# =========================
# Gráfico 1: Evolución de MRR
# =========================
st.subheader("Evolución de MRR")
line_mrr = alt.Chart(filt).mark_line(point=True).encode(
    x=alt.X('Date:T', title='Mes'),
    y=alt.Y('Total MRR (€):Q', title='MRR (€)'),
    tooltip=['Date:T','Total MRR (€):Q','New MRR (€):Q','Expansion MRR (inferred €):Q','Churned MRR (€):Q','Downgraded MRR (inferred €):Q']
).properties(height=300)
st.altair_chart(line_mrr, use_container_width=True)

# =========================
# Gráfico 2: Net New MRR (con filtros)
# =========================
st.subheader("NET NEW MRR por componentes")
components = {
    "New MRR (€)": st.checkbox("New", value=True),
    "Expansion MRR (inferred €)": st.checkbox("Expansion (inferred)", value=True),
    "Churned MRR (€)": st.checkbox("Churned", value=True),
    "Downgraded MRR (inferred €)": st.checkbox("Downgrade (inferred)", value=True)
}
stack_df = filt.melt(
    id_vars=["Date"],
    value_vars=[k for k, v in components.items() if v],
    var_name="Tipo", value_name="€"
)
bars = alt.Chart(stack_df).mark_bar().encode(
    x=alt.X('Date:T', title='Mes'),
    y=alt.Y('sum(€):Q', title='€'),
    color=alt.Color('Tipo:N'),
    tooltip=['Date:T','Tipo:N','€:Q']
).properties(height=300)
st.altair_chart(bars, use_container_width=True)

st.divider()

# =========================
# Tabla por plan
# =========================
plans = comp[comp["Date"].dt.year.isin(sel_years)].copy()
last_by_plan = plans.sort_values("Date").groupby("Plan").tail(1)
tot_mrr = last_by_plan["MRR (€)"].sum()
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

# =========================
# Cohorts (aprox. FIFO) — HEATMAP visual con m0=100% y meses sin cohortes
# =========================
st.subheader("Cohorts por mes de alta — Heatmap (aprox. FIFO)")

# Selector de plan y horizonte
plan_opts = ["(Todos)"] + sorted(df_prices["Plan"].astype(str).unique().tolist())
plan_for_cohorts = st.selectbox("Plan para cohorts", plan_opts, index=0)
horizon = st.slider("Horizonte (meses)", 6, 24, 12)

# Base de datos para cohorts (solo las columnas necesarias)
base = df_data[["Date", "Plan", "New Customers", "Lost Customers"]].copy()
if plan_for_cohorts != "(Todos)":
    base = base[base["Plan"] == plan_for_cohorts]

# --- FIFO corregido: m0 siempre 100% (no restar churn del mismo mes de alta) ---
def _fifo_retention_rows(df):
    df = df.sort_values(["Plan", "Date"]).reset_index(drop=True)
    rows = []
    for plan, g in df.groupby("Plan", sort=False):
        queue = []          # lista de [cohort_month, remaining]
        initial_map = {}    # tamaño inicial por cohorte (mes alta)
        for month, gm in g.groupby("Date", sort=True):
            new_c = int(gm["New Customers"].sum())
            lost_c = int(gm["Lost Customers"].sum())

            # 1) Añadir cohorte nuevo (si lo hay)
            if new_c > 0:
                queue.append([month, new_c])
                initial_map[month] = new_c

            # 2) Aplicar bajas SOLO a cohortes previas (no al cohort creado este mismo mes)
            remaining_to_remove = lost_c
            qi = 0
            while remaining_to_remove > 0 and qi < len(queue):
                cohort_month, remaining = queue[qi]
                if cohort_month == month:   # no tocar el cohort del mes actual
                    qi += 1
                    continue
                take = min(remaining, remaining_to_remove)
                queue[qi][1] -= take
                remaining_to_remove -= take
                if queue[qi][1] == 0:
                    qi += 1
            queue = [q for q in queue if q[1] > 0]

            # 3) Registrar estado de todas las cohortes vivas en este mes
            for cohort_month, remaining in queue:
                rows.append({
                    "Plan": plan,
                    "Cohort": cohort_month,
                    "Month": month,
                    "Remaining": remaining,
                    "Initial": initial_map.get(cohort_month, np.nan)
                })
    if not rows:
        return pd.DataFrame()
    res = pd.DataFrame(rows)
    res["Age (months)"] = (
        (res["Month"].dt.year - res["Cohort"].dt.year) * 12 +
        (res["Month"].dt.month - res["Cohort"].dt.month)
    ).astype(int)
    res["Retention %"] = res["Remaining"] / res["Initial"] * 100.0
    return res

res = _fifo_retention_rows(base)

if res.empty:
    st.info("No hay datos suficientes para construir cohorts con la selección actual.")
else:
    # --- Construir pivot TOTAL usando suma global por cohorte (m0 = 100%) ---
    # Remaining total por (Cohort, Age)
    total_rem = (res.groupby(["Cohort", "Age (months)"])["Remaining"]
                   .sum().reset_index())
    # Initial total por Cohort
    total_init = res.groupby("Cohort")["Initial"].sum()
    total_rem["Initial_total"] = total_rem["Cohort"].map(total_init)
    total_rem = total_rem[total_rem["Initial_total"] > 0]
    total_rem["Retention %"] = (total_rem["Remaining"] / total_rem["Initial_total"]) * 100.0
    pivot_total = total_rem.pivot(index="Cohort", columns="Age (months)", values="Retention %").sort_index()

    # --- Asegurar filas para meses SIN cohortes (mostrar guiones) ---
    all_months = pd.period_range(base["Date"].min().to_period("M"), base["Date"].max().to_period("M"), freq="M").to_timestamp()
    pivot_total = pivot_total.reindex(all_months)  # añade filas vacías para meses sin altas

    # --- Limitar al horizonte y preparar datos largos para heatmap ---
    age_cols = [c for c in pivot_total.columns if isinstance(c, (int, np.integer)) and c >= 0]
    age_cols = sorted(age_cols)[:horizon + 1]  # m0..mN
    pivot_show = pivot_total[age_cols].copy()

    # Para tooltip bonito (— en NaN) y etiquetas de fila
    df_long = pivot_show.reset_index().rename(columns={"index": "Cohort"})
    df_long["CohortLabel"] = df_long["Cohort"].dt.strftime("%Y-%m")
    df_long = df_long.melt(id_vars=["Cohort", "CohortLabel"], value_vars=age_cols,
                           var_name="Age", value_name="Retention")
    df_long["RetentionStr"] = df_long["Retention"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "—")

    # Detectar meses sin cohortes (toda la fila NaN)
    months_without_cohort = pivot_show.index[pivot_show.isna().all(axis=1)]
    months_without_cohort_lbl = [d.strftime("%Y-%m") for d in months_without_cohort]

    # --- Heatmap (Altair) ---
    heat = alt.Chart(df_long).mark_rect().encode(
        x=alt.X('Age:O', title='Edad del cohort (meses)'),
        y=alt.Y('CohortLabel:N', title='Mes de alta', sort='-y'),
        color=alt.Color('Retention:Q', title='Retención %', scale=alt.Scale(domain=[0, 100])),
        tooltip=[
            alt.Tooltip('CohortLabel:N', title='Cohort'),
            alt.Tooltip('Age:O', title='Edad (m)'),
            alt.Tooltip('RetentionStr:N', title='Retención %')
        ]
    ).properties(height=320)

    st.altair_chart(heat, use_container_width=True)
    st.caption("Cada fila es el **mes de alta**. Las columnas son la **edad del cohort** (m0..mN). El color indica la **supervivencia (%)**. "
               "En **(Todos)** se usa suma global (garantiza m0=100%). Los meses **sin cohortes** aparecen como filas vacías (celdas ‘—’ en tooltip).")

    if months_without_cohort_lbl:
        st.markdown(
            "Meses sin cohortes (sin altas): " +
            ", ".join(f"`{m}`" for m in months_without_cohort_lbl)
        )

st.divider()

# =========================
# Valoración
# =========================
st.subheader("Valoración — total y por plan")

mrr_last = mrr_now
arr_last = arr_now
mult_slider = st.slider("Múltiplo xARR (ajustable según sector)", 4.0, 25.0, float(base_mult), 0.5)
valuation_total = arr_last * mult_slider if pd.notna(arr_last) else np.nan
st.metric("Valoración total (sector)", fmt_money(valuation_total))

per_plan_val = table[["Plan","ARR (€)","Múltiplo plan (x ARR)"]].copy()
per_plan_val["Valoración (€)"] = per_plan_val["ARR (€)"] * per_plan_val["Múltiplo plan (x ARR)"]
st.dataframe(per_plan_val.sort_values("Valoración (€)", ascending=False), use_container_width=True)

# =========================
# Simulador 3–5 años
# =========================
st.subheader("Simulador de valoración (3–5 años)")
c1, c2, c3 = st.columns(3)
with c1:
    sim_arr0 = st.number_input("ARR actual (€)", value=float(arr_last) if pd.notna(arr_last) else 0.0, step=1000.0, format="%.2f")
with c2:
    sim_growth_m = st.slider("Crecimiento mensual (%)", 0.0, 30.0, 5.0, 0.1)
with c3:
    sim_churn_m = st.slider("Churn mensual (%)", 0.0, 15.0, 2.0, 0.1)

net_g = (sim_growth_m - sim_churn_m) / 100.0
years = [3,4,5]
proj = []
for Y in years:
    arrY = sim_arr0 * ((1 + net_g) ** (12 * Y))
    # Ajuste heurístico del múltiplo por crecimiento/churn (opcional):
    growth_a = ((1 + sim_growth_m/100.0)**12 - 1)
    churn_a  = ((1 + sim_churn_m/100.0)**12 - 1)
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

st.success("Consejo: rellena CAC para obtener LTV/CAC y ajusta el **margen bruto** para un LTV realista.")
