import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as make_subplots
import pulp
import warnings
from typing import Optional

# --- Pagina Configuratie ---
st.set_page_config(layout="wide", page_title="Batterijsimulatie Dashboard")

# --- Functies voor de Analyse (met caching voor snelheid) ---

@st.cache_data
def analyze_data(df_raw, battery_config):
    # Deze functie bevat alle logica uit uw notebook.
    # Het neemt de onbewerkte data en de batterij-instellingen als input
    # en geeft een DataFrame met alle resultaten terug.

    # Helper functie voor kolomnormalisatie
    def _rename_aliases(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.copy()
        lowmap = {str(c).strip().lower(): c for c in df_out.columns}
        def pick(*aliases, required=False):
            for a in aliases:
                if a in lowmap: return lowmap[a]
            if required: raise KeyError(f"Missing any of columns: {aliases}")
            return None

        col_date    = pick("date", "datetime", "timestamp", required=True)
        col_import  = pick("import_kwh", "import", "grid import (kwh)", "grid_import_kwh", required=True)
        col_inject  = pick("injection_kwh", "injection", "export_kwh", "export (kwh)")
        col_pv      = pick("pv_kwh", "zonne-opbrengst (kwh)", "pv", "solar_kwh")
        col_cons    = pick("consumption_kwh", "verbruik (kwh)", "consumption")
        col_belpex  = pick("belpex", required=True)

        rename = {
            col_date: "Date", col_import: "Import_KWh",
            col_belpex: "BELPEX"
        }
        if col_inject: rename[col_inject] = "Injection_KWh"
        if col_pv: rename[col_pv] = "PV_KWh"
        if col_cons: rename[col_cons] = "Consumption_KWh"
        
        return df_out.rename(columns=rename)

    df = _rename_aliases(df_raw)
    
    # ... (Alle data processing en berekeningen uit uw notebook) ...
    # Types & sort
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if "Injection_KWh" not in df.columns:
        if {"PV_KWh", "Consumption_KWh"}.issubset(df.columns):
            df["Injection_KWh"] = (df["PV_KWh"] - df["Consumption_KWh"]).clip(lower=0.0)
        else:
            df["Injection_KWh"] = 0.0
    
    # Prijsberekening
    belpex_price_eur_kwh = pd.to_numeric(df["BELPEX"], errors="coerce").fillna(0.0) / 1000.0
    df["Prijs_verbruik"] = battery_config["belpex_buy_offset"] + (battery_config["belpex_buy_factor"] * belpex_price_eur_kwh)
    df["Prijs_injectie"] = battery_config["belpex_sell_offset"] + (battery_config["belpex_sell_factor"] * belpex_price_eur_kwh)

    # Step detection
    if len(df) > 1:
        dt_s = df["Date"].diff().dropna().dt.total_seconds().mode()
        DT_HOURS = float(dt_s.iloc[0]/3600.0) if len(dt_s) else 1.0
    else:
        DT_HOURS = 1.0

    CHARGE_CAP_STEP_KWH = battery_config["charge_cap_per_hour"] * DT_HOURS
    DISCHARGE_CAP_STEP_KWH = battery_config["discharge_cap_per_hour"] * DT_HOURS

    # Baseline bill
    price_buy  = pd.to_numeric(df["Prijs_verbruik"], errors="coerce").fillna(0.0)
    price_sell = pd.to_numeric(df["Prijs_injectie"], errors="coerce").fillna(0.0)
    imp0 = pd.to_numeric(df["Import_KWh"], errors="coerce").fillna(0.0)
    inj0 = pd.to_numeric(df["Injection_KWh"], errors="coerce").fillna(0.0)
    df['baseline_cost_per_step'] = imp0 * price_buy - inj0 * price_sell

    # Auto-consumption pass
    N = len(df)
    chg_pv_auto, dis_auto, grid_after, export_after, soc_auto = (np.zeros(N) for _ in range(5))
    soc = battery_config["start_soc"]
    for t in range(N):
        headroom_out = battery_config["capacity"] - soc
        take = min(inj0[t], CHARGE_CAP_STEP_KWH / max(battery_config["rte"], 1e-9), headroom_out / max(battery_config["rte"], 1e-9))
        chg_pv_auto[t] = take
        soc += battery_config["rte"] * take
        avail = max(0.0, soc - battery_config["reserve_soc"])
        d = min(imp0[t], DISCHARGE_CAP_STEP_KWH, avail)
        dis_auto[t] = d
        soc -= d
        grid_after[t] = max(0.0, imp0[t] - d)
        export_after[t] = max(0.0, inj0[t] - take)
        soc = min(max(soc, 0.0), battery_config["capacity"])
        soc_auto[t] = soc
    
    df["Charge_from_PV_auto_kWh"] = chg_pv_auto
    df["Discharge_auto2_kWh"] = dis_auto
    df["Grid_buy_after_auto_kWh"] = grid_after
    df["Export_after_auto_kWh"] = export_after
    df["SoC_auto2_KWh"] = soc_auto
    df['auto_cost_per_step'] = df["Grid_buy_after_auto_kWh"] * price_buy - df["Export_after_auto_kWh"] * price_sell

    # MILP pass
    price_np = price_buy.to_numpy(float)
    price_inj_np = price_sell.to_numpy(float)
    import_arr = imp0.to_numpy(float)
    
    df['Maand_Jaar'] = df['Date'].dt.to_period('M')
    piekverbruik_maand = df.groupby('Maand_Jaar')['Import_KWh'].max().to_dict()
    df['piekverbruik_maand'] = df['Maand_Jaar'].map(piekverbruik_maand)
    peak_cap_arr = df['piekverbruik_maand'].to_numpy(float)

    steps_per_day = int(24 / DT_HOURS)
    charge_candidates, disch_candidates = set(), set()
    for t_start in range(0, N, steps_per_day):
        t_end = min(t_start + steps_per_day, N)
        daily_slice = df.iloc[t_start:t_end].copy()
        daily_slice['global_index'] = daily_slice.index
        cheapest_grid_charge = daily_slice.sort_values(by="Prijs_verbruik").head(int(battery_config["max_cand_charge"]/DT_HOURS))
        charge_candidates.update(cheapest_grid_charge['global_index'].tolist())
        richest_pv_charge = daily_slice[daily_slice["Injection_KWh"] > 0].sort_values(by="Prijs_injectie", ascending=False).head(int(battery_config["max_cand_charge"]/DT_HOURS))
        charge_candidates.update(richest_pv_charge['global_index'].tolist())
        richest_import = daily_slice[daily_slice["Import_KWh"] > 0].sort_values(by="Prijs_injectie", ascending=False).head(int(battery_config["max_cand_disch"]/DT_HOURS))
        disch_candidates.update(richest_import['global_index'].tolist())

    prob = pulp.LpProblem("battery_profit_max", pulp.LpMaximize)
    charge_g = pulp.LpVariable.dicts("charge_grid", range(N), lowBound=0)
    charge_p = pulp.LpVariable.dicts("charge_pv", range(N), lowBound=0)
    discharge_g = pulp.LpVariable.dicts("discharge_grid", range(N), lowBound=0)
    discharge_l = pulp.LpVariable.dicts("discharge_load", range(N), lowBound=0)
    soc_t = pulp.LpVariable.dicts("soc", range(N + 1), lowBound=0, upBound=battery_config["capacity"])
    pv_to_grid = pulp.LpVariable.dicts("pv_to_grid", range(N), lowBound=0)
    grid_to_load = pulp.LpVariable.dicts("grid_to_load", range(N), lowBound=0)

    prob += pulp.lpSum((price_inj_np[t] * (discharge_g[t] + pv_to_grid[t])) - (price_np[t] * (grid_to_load[t] + charge_g[t])) for t in range(N))
    for t in range(N):
        prob += soc_t[t+1] == soc_t[t] + battery_config["rte"] * (charge_g[t] + charge_p[t]) - (discharge_g[t] + discharge_l[t])
        prob += charge_g[t] + charge_p[t] <= CHARGE_CAP_STEP_KWH
        prob += discharge_g[t] + discharge_l[t] <= DISCHARGE_CAP_STEP_KWH
        prob += soc_t[t] >= battery_config["reserve_soc"]
        if "Injection_KWh" in df.columns: prob += charge_p[t] + pv_to_grid[t] == df['Injection_KWh'].iloc[t]
        else: prob += charge_p[t] == 0; prob += pv_to_grid[t] == 0
        prob += grid_to_load[t] + charge_g[t] <= peak_cap_arr[t]
        prob += pv_to_grid[t] + discharge_g[t] <= peak_cap_arr[t]
        prob += grid_to_load[t] + discharge_l[t] >= import_arr[t]
        prob += grid_to_load[t] <= import_arr[t]
        if t not in charge_candidates: prob += charge_g[t] == 0; prob += charge_p[t] == 0
        if t not in disch_candidates: prob += discharge_g[t] == 0
    prob += soc_t[0] == battery_config["start_soc"]; prob += soc_t[N] == battery_config["start_soc"]
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    df["Charge_from_grid_da_kWh"] = [pulp.value(v) for v in charge_g.values()]
    df["Charge_from_PV_da_kWh"]    = [pulp.value(v) for v in charge_p.values()]
    df["Discharge_da_kWh"]         = [pulp.value(discharge_g[t]) + pulp.value(discharge_l[t]) for t in range(N)]
    df["SoC_da_KWh"]               = [pulp.value(soc_t[t+1]) for t in range(N)]
    df["Grid_total_da_KWh"]        = [pulp.value(grid_to_load[t]) + pulp.value(charge_g[t]) for t in range(N)]
    df["Export_after_da_KWh"]      = [pulp.value(discharge_g[t]) + pulp.value(pv_to_grid[t]) for t in range(N)]
    df['da_cost_per_step'] = df["Grid_total_da_KWh"] * price_buy - df["Export_after_da_KWh"] * price_sell
    return df

# --- Functies voor Grafieken ---
def make_week_picker_fig(wk_df, title):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=wk_df["week"], y=wk_df["savings_da"], name="Besparing Day-ahead (€)", opacity=0.7))
    fig.update_layout(title=title, barmode="group", height=300, margin=dict(l=40,r=20,t=50,b=40), xaxis_title="ISO Week", yaxis_title="€")
    return fig

def make_week_series_fig(dd, scenario, battery_capacity):
    if scenario == "auto":
        y_import_after = dd["Grid_buy_after_auto_kWh"]
        y_export_after = -dd["Export_after_auto_kWh"]
        soc = dd.get("SoC_auto2_KWh", pd.Series(0.0, index=dd.index))
        title = f"Weekdetail (Auto-consumptie)"
    else:
        y_import_after = dd["Grid_total_da_KWh"]
        y_export_after = -dd["Export_after_da_KWh"]
        soc = dd.get("SoC_da_KWh", pd.Series(0.0, index=dd.index))
        title = f"Weekdetail (Day-ahead)"

    fig = make_subplots.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, specs=[[{"secondary_y": False}], [{"secondary_y": True}]])
    
    # Energiestromen
    fig.add_trace(go.Bar(x=dd["Date"], y=dd["Import_KWh"], name="Import (baseline)", marker_color='rgba(255,0,0,0.5)'), row=1, col=1)
    fig.add_trace(go.Bar(x=dd["Date"], y=-dd["Injection_KWh"], name="Injectie (baseline)", marker_color='rgba(0,128,0,0.5)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd["Date"], y=y_import_after, mode="lines", name=f"Import ({scenario})", line=dict(color='red', width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd["Date"], y=y_export_after, mode="lines", name=f"Injectie ({scenario})", line=dict(color='green', width=2.5)), row=1, col=1)

    # Prijzen en SoC
    fig.add_trace(go.Scatter(x=dd["Date"], y=dd["Prijs_verbruik"], mode="lines", name="Prijs Afname (€/kWh)", line=dict(color='red')), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=dd["Date"], y=dd["Prijs_injectie"], mode="lines", name="Prijs Injectie (€/kWh)", line=dict(color='green')), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=dd["Date"], y=soc, mode="lines", name="Batterij SoC (kWh)", line=dict(dash="dash", color='purple')), row=2, col=1, secondary_y=True)

    fig.update_layout(title=title, hovermode="x unified", height=500, margin=dict(l=50, r=60, t=50, b=50), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="kWh", row=1, col=1)
    fig.update_yaxes(title_text="€/kWh", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="SoC (kWh)", row=2, col=1, secondary_y=True, range=[0, battery_capacity])
    
    return fig


# --- Streamlit App Layout ---

st.title("🔋 Batterijsimulatie Dashboard")

# --- Sidebar voor Instellingen ---
with st.sidebar:
    st.header("⚙️ Configuratie")
    
    st.subheader("Data Input")
    uploaded_file = st.file_uploader("Kies een Excel-bestand (.xlsx)", type="xlsx")
    sheet_name = st.text_input("Naam van het werkblad", "Load profiles")

    st.subheader("Batterij-instellingen")
    BATTERY_CAP_KWH = st.number_input("Batterijcapaciteit (kWh)", min_value=1.0, value=300.0, step=10.0)
    CHARGE_HOURS_TO_FULL = st.number_input("Laadtijd van leeg naar vol (uur)", min_value=0.5, value=2.0, step=0.5)
    DISCHARGE_HOURS_TO_EMPTY = st.number_input("Ontlaadtijd van vol naar leeg (uur)", min_value=0.5, value=2.0, step=0.5)
    RTE = st.slider("Round-trip efficiency (%)", min_value=80, max_value=100, value=100) / 100.0

    st.subheader("Prijsmodel (op basis van BELPEX)")
    BELPEX_BUY_PRICE_OFFSET_EUR_KWH = st.number_input("Toeslag op aankoopprijs (€/kWh)", value=0.029, format="%.3f")
    BELPEX_SELL_PRICE_OFFSET_EUR_KWH = st.number_input("Afslag op verkoopprijs (€/kWh)", value=-0.020, format="%.3f")

battery_config = {
    "capacity": BATTERY_CAP_KWH,
    "charge_cap_per_hour": BATTERY_CAP_KWH / max(CHARGE_HOURS_TO_FULL, 1e-9),
    "discharge_cap_per_hour": BATTERY_CAP_KWH / max(DISCHARGE_HOURS_TO_EMPTY, 1e-9),
    "rte": RTE,
    "start_soc": 0.0,
    "reserve_soc": 0.0,
    "belpex_buy_offset": BELPEX_BUY_PRICE_OFFSET_EUR_KWH,
    "belpex_buy_factor": 1.0,
    "belpex_sell_offset": BELPEX_SELL_PRICE_OFFSET_EUR_KWH,
    "belpex_sell_factor": 1.0,
    "max_cand_charge": 8,
    "max_cand_disch": 8
}

# --- Hoofdpagina ---
if uploaded_file is not None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        
        results_df = analyze_data(df_raw, battery_config)
        
        baseline_cost = results_df['baseline_cost_per_step'].sum()
        auto_cost = results_df['auto_cost_per_step'].sum()
        dayahead_cost = results_df['da_cost_per_step'].sum()
        savings_auto = baseline_cost - auto_cost
        savings_da = baseline_cost - dayahead_cost

        st.header("📈 Financiële Samenvatting")
        col1, col2, col3 = st.columns(3)
        col1.metric("Kosten Zonder Batterij", f"€ {baseline_cost:,.0f}")
        col2.metric("Besparing Auto-consumptie", f"€ {savings_auto:,.0f}")
        col3.metric("Besparing Day-ahead", f"€ {savings_da:,.0f}")
        
        st.header("📊 Wekelijkse Besparingen")
        
        results_df["ISO_Week"] = results_df["Date"].dt.isocalendar().week.astype(int)
        results_df["ISO_Year"] = results_df["Date"].dt.isocalendar().year.astype(int)
        week_keys = results_df["ISO_Year"].astype(str) + "-W" + results_df["ISO_Week"].astype(str).str.zfill(2)

        base_week  = results_df.groupby(week_keys)['baseline_cost_per_step'].sum()
        auto_week  = results_df.groupby(week_keys)['auto_cost_per_step'].sum()
        da_week    = results_df.groupby(week_keys)['da_cost_per_step'].sum()
        
        wk_index = sorted(base_week.index.unique())
        wk_df = pd.DataFrame({
            "week": wk_index, "baseline": base_week.reindex(wk_index).fillna(0.0).values,
            "auto": auto_week.reindex(wk_index).fillna(0.0).values,
            "da": da_week.reindex(wk_index).fillna(0.0).values,
        })
        wk_df["savings_auto"] = wk_df["baseline"] - wk_df["auto"]
        wk_df["savings_da"]   = wk_df["baseline"] - wk_df["da"]
        
        st.plotly_chart(make_week_picker_fig(wk_df, "Wekelijkse besparing (Day-ahead vs. Baseline)"), use_container_width=True)

        st.header("🗓️ Weekdetail Analyse")
        selected_week = st.selectbox("Kies een week voor detailweergave", options=wk_index)
        
        if selected_week:
            dd = results_df[week_keys == selected_week].copy()
            scenario = st.radio("Kies een scenario om te tonen:", ('Day-ahead', 'Auto-consumptie'), horizontal=True, key='scenario_radio')
            scenario_map = {'Day-ahead': 'da', 'Auto-consumptie': 'auto'}
            fig = make_week_series_fig(dd, scenario_map[scenario], BATTERY_CAP_KWH)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Bekijk de volledige resultatentabel"):
            st.dataframe(results_df)

    except Exception as e:
        st.error(f"Er is een fout opgetreden bij het verwerken van het bestand: {e}")
        st.warning(f"Controleer of de sheet '{sheet_name}' bestaat en de juiste kolommen bevat (Date, Import_KWh, Injection_KWh, PV_KWh, BELPEX).")

else:
    st.info("Upload een Excel-bestand en configureer de instellingen in de zijbalk om de simulatie te starten.")