import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import pulp
import plotly.graph_objects as go
import plotly.subplots as make_subplots
import os
import altair as alt

# --- Pagina Configuratie ---
st.set_page_config(layout="wide")
# --- Titel met Logo en Icoon ---
col1, col2, col3 = st.columns([1, 4, 1]) # Maak 3 kolommen met verhouding 1:4:1

with col1:
    logo_path = "kiozenergy_logo.jpeg"
    if os.path.exists(logo_path):
        st.image(logo_path, width=150) # Toon logo in de eerste kolom

with col2:
    st.title("Optimalisatie Day-Ahead en Peak Shaving") # Titel in de middelste, breedste kolom

with col3:
    # Gebruik markdown om het icoon groot en rechts uit te lijnen
    st.markdown("<h1 style='text-align: right;'>🔋</h1>", unsafe_allow_html=True)

# --- Functie Definities ---
# --- Gecachete Analyse Functie ---
@st.cache_data
def run_full_analysis(
    data_path, battery_cap, laadsnelheid, rte, verbruiker_type, 
    buy_offset_groot, buy_offset_klein, sell_offset, peak_cost
):
    """
    Voert de volledige dataverwerking en optimalisatie uit.
    """
    
    # --- AANGEPAST: Parameters hardcoded op 0 ---
    start_soc = 0.0
    reserve_soc = 0.0
    
    # --- AANGEPAST: Laadsnelheid voor zowel laden als ontladen ---
    charge_hours = laadsnelheid
    discharge_hours = laadsnelheid
    
    # Bepaal de meerkost op basis van het verbruikerstype
    meerkost_kleinverbruiker = buy_offset_klein if verbruiker_type == "Nee" else 0
    buy_offset = buy_offset_groot + meerkost_kleinverbruiker


    # (De rest van de functie blijft grotendeels hetzelfde)
    def _rename_aliases(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.copy()
        lowmap = {str(c).strip().lower(): c for c in df_out.columns}
        def pick(*aliases, required=False):
            for a in aliases:
                if a in lowmap: return lowmap[a]
            if required: raise KeyError(f"Ontbrekende kolom: {aliases}")
            return None
        col_date = pick("date", "datetime", "timestamp", required=True)
        col_import = pick("import_kwh", "import", "grid import (kwh)", "grid_import_kwh", required=True)
        col_inject = pick("injection_kwh", "injection", "export_kwh", "export (kwh)")
        col_pv = pick("pv_kwh", "zonne-opbrengst (kwh)", "pv", "solar_kwh")
        col_cons = pick("consumption_kwh", "verbruik (kwh)", "consumption")
        col_belpex = pick("belpex", required=True)
        rename = {col_date: "Date", col_import: "Import_KWh", col_belpex: "BELPEX"}
        if col_inject: rename[col_inject] = "Injection_KWh"
        if col_pv: rename[col_pv] = "PV_KWh"
        if col_cons: rename[col_cons] = "Consumption_KWh"
        df_out = df_out.rename(columns=rename)
        return df_out

    raw_df = pd.read_excel(data_path, sheet_name="Load profiles")
    df = _rename_aliases(raw_df)
    
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M")
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.drop_duplicates(subset=['Date'], keep='first').reset_index(drop=True)

    if "Injection_KWh" not in df.columns:
        if {"PV_KWh", "Consumption_KWh"}.issubset(df.columns):
            df["Injection_KWh"] = (df["PV_KWh"] - df["Consumption_KWh"]).clip(lower=0.0)
        else:
            df["Injection_KWh"] = 0.0
    
    belpex_price_eur_kwh = pd.to_numeric(df["BELPEX"], errors="coerce").fillna(0.0) / 1000.0
    # AANGEPAST: Gebruikt de nieuwe buy_offset logica
    df["Prijs_verbruik"] = buy_offset + (1.00 * belpex_price_eur_kwh)
    df["Prijs_injectie"] = sell_offset + (1.00 * belpex_price_eur_kwh)

    if len(df) > 1:
        dt_s = df["Date"].diff().dropna().dt.total_seconds().mode()
        DT_HOURS = float(dt_s.iloc[0]/3600.0) if not dt_s.empty else 1.0
    else:
        DT_HOURS = 1.0

    charge_cap_hour = battery_cap / max(charge_hours, 1e-9)
    discharge_cap_hour = battery_cap / max(discharge_hours, 1e-9)
    charge_cap_step = charge_cap_hour * DT_HOURS
    discharge_cap_step = discharge_cap_hour * DT_HOURS
    
    price_buy  = pd.to_numeric(df["Prijs_verbruik"], errors="coerce").fillna(0.0)
    price_sell = pd.to_numeric(df["Prijs_injectie"], errors="coerce").fillna(0.0)
    imp0 = pd.to_numeric(df["Import_KWh"], errors="coerce").fillna(0.0)
    inj0 = pd.to_numeric(df["Injection_KWh"], errors="coerce").fillna(0.0)
    N = len(df)
    
    chg_pv_auto, dis_auto, grid_after, export_after, soc_auto = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    soc = start_soc
    for t in range(N):
        headroom_out = battery_cap - soc
        take = min(inj0[t], charge_cap_step/max(rte,1e-9), headroom_out/max(rte,1e-9))
        chg_pv_auto[t] = take
        soc += rte * take
        avail = max(0.0, soc - reserve_soc)
        d = min(imp0[t], discharge_cap_step, avail)
        dis_auto[t] = d
        soc -= d
        grid_after[t] = max(0.0, imp0[t] - d)
        export_after[t] = max(0.0, inj0[t] - take)
        soc = min(max(soc, 0.0), battery_cap)
        soc_auto[t] = soc
    df["Charge_from_PV_auto_kWh"] = chg_pv_auto
    df["Discharge_auto2_kWh"] = dis_auto
    df["Grid_buy_after_auto_kWh"] = grid_after
    df["Export_after_auto_kWh"] = export_after
    df["SoC_auto2_KWh"] = soc_auto

    prob = pulp.LpProblem("battery_profit_max", pulp.LpMaximize)
    df['Maand_Jaar'] = df['Date'].dt.to_period('M')
    unique_months = sorted(df['Maand_Jaar'].unique())
    month_of_t = [df['Maand_Jaar'].iloc[t] for t in range(N)]
    
    charge_g = pulp.LpVariable.dicts("charge_grid", range(N), lowBound=0)
    charge_p = pulp.LpVariable.dicts("charge_pv", range(N), lowBound=0)
    discharge_g = pulp.LpVariable.dicts("discharge_grid", range(N), lowBound=0)
    discharge_l = pulp.LpVariable.dicts("discharge_load", range(N), lowBound=0)
    soc_t = pulp.LpVariable.dicts("soc", range(N + 1), lowBound=0, upBound=battery_cap)
    pv_to_grid = pulp.LpVariable.dicts("pv_to_grid", range(N), lowBound=0)
    grid_to_load = pulp.LpVariable.dicts("grid_to_load", range(N), lowBound=0)
    max_original_injection = df['Injection_KWh'].max() * 1.01
    peak_power_month_kw = pulp.LpVariable.dicts("peak_power_month_kw", unique_months, lowBound=0)
    grid_total_power_kw = pulp.LpVariable.dicts("grid_total_power_kw", range(N), lowBound=0)
    
    prob += pulp.lpSum( (price_sell[t] * (discharge_g[t] + pv_to_grid[t])) - (price_buy[t] * (grid_to_load[t] + charge_g[t])) for t in range(N) ) - \
            pulp.lpSum( peak_cost * peak_power_month_kw[m] for m in unique_months )

    for t in range(N):
        prob += soc_t[t+1] == soc_t[t] + rte * (charge_g[t] + charge_p[t]) - (discharge_g[t] + discharge_l[t])
        prob += charge_g[t] + charge_p[t] <= charge_cap_step
        prob += discharge_g[t] + discharge_l[t] <= discharge_cap_step
        prob += soc_t[t] >= reserve_soc
        prob += charge_p[t] + pv_to_grid[t] == inj0[t]
        prob += grid_total_power_kw[t] == (grid_to_load[t] + charge_g[t]) / DT_HOURS
        prob += peak_power_month_kw[month_of_t[t]] >= grid_total_power_kw[t]
        prob += discharge_g[t] + pv_to_grid[t] <= max_original_injection
        prob += grid_to_load[t] + discharge_l[t] == imp0[t]

    prob += soc_t[0] == start_soc
    prob += soc_t[N] == start_soc
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    df["Charge_from_grid_da_kWh"] = [pulp.value(v) for v in charge_g.values()]
    df["Charge_from_PV_da_kWh"] = [pulp.value(v) for v in charge_p.values()]
    df["Discharge_da_kWh"] = [pulp.value(discharge_g[t]) + pulp.value(discharge_l[t]) for t in range(N)]
    df["SoC_da_KWh"] = [pulp.value(soc_t[t+1]) for t in range(N)]
    df["Grid_total_da_KWh"] = [pulp.value(grid_to_load[t]) + pulp.value(charge_g[t]) for t in range(N)]
    df["Export_after_da_KWh"] = [pulp.value(discharge_g[t]) + pulp.value(pv_to_grid[t]) for t in range(N)]
    df["Discharge_to_load_da_kWh"] = [pulp.value(v) for v in discharge_l.values()]
    df["Discharge_to_grid_da_kWh"] = [pulp.value(v) for v in discharge_g.values()]
    
    optimized_peaks = {str(m): pulp.value(peak_power_month_kw[m]) for m in unique_months}
    
    return df, DT_HOURS, unique_months, optimized_peaks

def _monthly_avg_block(df, y_cols, labels, title, soc_col=None, soc_label="Battery SoC (kWh)", soc_divisor=1):
    dfx = df.copy()
    dfx["Date"] = pd.to_datetime(dfx["Date"]).dt.round('15min')
    dfx["HourOfDay"] = dfx["Date"].dt.hour + dfx["Date"].dt.minute / 60.0
    dfx["Month"] = dfx["Date"].dt.to_period("M")
    months = sorted(dfx["Month"].unique())
    if not months:
        st.write("Geen data om te plotten voor de geselecteerde periode.")
        return None
    x_all, series_all, soc_all = [], [[] for _ in y_cols], []
    xticks, xlabels, seps, pos = [], [], [], 0
    for m in months:
        sub = dfx[dfx["Month"] == m].copy()
        sub["H"], sub["M"] = sub["Date"].dt.hour, sub["Date"].dt.minute
        grp = sub.groupby(["H", "M"]).mean(numeric_only=True)
        if grp.empty: continue
        idx = sorted(grp.index, key=lambda z: (z[0], z[1]))
        prof = grp.loc[idx]
        steps = len(prof)
        x_seg = np.arange(steps) + pos
        x_all.append(x_seg)
        for j, col in enumerate(y_cols):
            series_all[j].append(prof[col].to_numpy() if col in prof.columns else np.zeros(steps))
        if soc_col and soc_col in prof.columns:
            soc_all.append(prof[soc_col].to_numpy() / soc_divisor)
        elif soc_col:
            soc_all.append(np.zeros(steps))
        seps.append(pos)
        xticks.append(pos + steps / 2)
        xlabels.append(m.to_timestamp().strftime("%b %Y"))
        pos += steps
    if not x_all:
        st.write("Geen data om te plotten.")
        return None
    x = np.concatenate(x_all)
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["tab:blue", "tab:red", "black", "tab:green", "tab:orange", "tab:cyan"]
    for j, (lbl, col) in enumerate(zip(labels, y_cols)):
        y = np.concatenate(series_all[j])
        ax.plot(x, y, label=lbl, color=colors[j % len(colors)], linewidth=1.8)
    if soc_col and soc_all:
        soc = np.concatenate(soc_all)
        soc_plot_color = "tab:purple"
        soc_display_label = f"{soc_label} / {soc_divisor}" if soc_divisor != 1 else soc_label
        ax.plot(x, soc, label=soc_display_label, color=soc_plot_color, linestyle="--", linewidth=1.6)
    ax.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
    for s in seps:
        ax.axvline(s - 0.5, color="#ddd", linewidth=0.8)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylabel("kWh")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig

def make_week_series_fig(dd, week_key, scenario, battery_cap):
    if dd.empty:
        return go.Figure().update_layout(title=f"Geen data voor week {week_key}")
    if scenario == "auto":
        y_import_after, y_export_after, soc = dd["Grid_buy_after_auto_kWh"], -dd["Export_after_auto_kWh"], dd.get("SoC_auto2_KWh", pd.Series(0.0, index=dd.index))
        title = f"Week {week_key} (Auto-consumptie)"
    else:
        y_import_after, y_export_after, soc = dd["Grid_total_da_KWh"], -dd["Export_after_da_KWh"], dd.get("SoC_da_KWh", pd.Series(0.0, index=dd.index))
        title = f"Week {week_key} (Day-ahead)"
    fig = make_subplots.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
                                      subplot_titles=("Energiestromen", "Prijzen en SoC"),
                                      specs=[[{"secondary_y": False}], [{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=dd["Date"], y=dd.get("PV_KWh", pd.Series(0.0, index=dd.index)), name="PV (kWh)", marker_color='gold'), row=1, col=1)
    fig.add_trace(go.Bar(x=dd["Date"], y=dd["Import_KWh"], name="Import (baseline)", marker_color='rgba(255,0,0,0.5)', opacity=0.5), row=1, col=1)
    fig.add_trace(go.Bar(x=dd["Date"], y=-dd["Injection_KWh"], name="Injectie (baseline)", marker_color='rgba(0,128,0,0.5)', opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd["Date"], y=y_import_after, mode="lines", name=f"Import ({scenario})", line=dict(color='red', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd["Date"], y=y_export_after, mode="lines", name=f"Injectie ({scenario})", line=dict(color='green', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd["Date"], y=dd["Prijs_verbruik"], mode="lines", name="Prijs afname (€/kWh)", line=dict(color='red')), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=dd["Date"], y=dd["Prijs_injectie"], mode="lines", name="Prijs injectie (€/kWh)", line=dict(color='green')), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=dd["Date"], y=soc, mode="lines", name="Batterij SoC (kWh)", line=dict(dash="dash", color='purple')), row=2, col=1, secondary_y=True)
    fig.update_layout(title=title, hovermode="x unified", height=600, margin=dict(l=50, r=60, t=80, b=50), legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(title_text="kWh", row=1, col=1)
    fig.update_yaxes(title_text="€/kWh", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="SoC (kWh)", row=2, col=1, secondary_y=True, range=[0, battery_cap * 1.05])
    return fig
# --- SIDEBAR: Keuzemenu voor bestandsbron ---
st.sidebar.header("Instellingen")
data_dir = "data"
os.makedirs(data_dir, exist_ok=True) 

# --- AANGEPAST: Positie van radio knoppen omgewisseld ---
source_choice = st.sidebar.radio("Kies een gegevensbron:", ('Selecteer een bestaand bestand', 'Upload een nieuw bestand'))
data_source = None

if source_choice == 'Selecteer een bestaand bestand':
    try:
        excel_files = [f for f in os.listdir(data_dir) if f.endswith(('.xlsx', '.xls'))]
        if not excel_files:
            st.sidebar.warning(f"Geen Excel-bestanden gevonden in de map '{data_dir}'. Upload eerst een bestand.")
        else:
            selected_file = st.sidebar.selectbox("Kies een bestand uit de 'data' map:", excel_files)
            data_source = os.path.join(data_dir, selected_file)
    except Exception as e:
        st.sidebar.error(f"Fout bij het lezen van de 'data' map: {e}")
else: # 'Upload een nieuw bestand'
    uploaded_file = st.sidebar.file_uploader("Kies een Excel-bestand", type=["xlsx", "xls"])
    if uploaded_file is not None:
        save_path = os.path.join(data_dir, uploaded_file.name)
        with open(save_path, "wb") as f: f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Bestand '{uploaded_file.name}' opgeslagen.")
        data_source = save_path

# --- AANGEPAST: Startknop direct onder de bestandskeuze ---
run_button = st.sidebar.button("Analyse Uitvoeren", type="primary")

st.sidebar.markdown("---") # Visuele scheidingslijn

# --- SIDEBAR: Parameter Inputs ---
BATTERY_CAP_KWH = st.sidebar.number_input("Batterijcapaciteit (kWh)", value=150.0, key="batt_cap")
# START_SOC en RESERVE_SOC zijn weggelaten

# --- AANGEPAST: Eén parameter voor laadsnelheid ---
LAADSNELHEID_UREN = st.sidebar.number_input("Laadsnelheid (uren)", value=2.0, min_value=0.1, key="laadsnelheid")

RTE = st.sidebar.slider("Round Trip Efficiency (RTE)", 0.0, 1.0, 0.88, key="rte")

# --- AANGEPAST: Logica voor cabine / kleinverbruiker ---
CABINE_AANWEZIG = st.sidebar.radio("Cabine aanwezig", ('Ja', 'Nee'), key="cabine")
BELPEX_BUY_PRICE_OFFSET_GROOT_EUR_KWH = st.sidebar.number_input("Belpex Aankoop Prijs Offset (€/kWh)", value=0.029, format="%.3f", key="buy_offset_groot")
MEERKOST_KLEINVERBRUIKER_EUR_KWH = st.sidebar.number_input("Meerkost kleinverbruiker (€/kWh)", value=0.119, format="%.3f", key="meerkost_klein")
BELPEX_SELL_PRICE_OFFSET_EUR_KWH = st.sidebar.number_input("Belpex Verkoop Prijs Offset (€/kWh)", value=-0.020, format="%.3f", key="sell_offset")
COST_PEAK_EUR_PER_KW_MONTH = st.sidebar.number_input("Kosten Piekvermogen (€/kW/maand)", value=5.0, key="peak_cost")


# --- STATE & ANALYSE TRIGGER ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

if run_button:
    if data_source:
        with st.spinner("Analyse wordt uitgevoerd... Dit kan even duren."):
            # --- AANGEPAST: Functie-aanroep met nieuwe parameters ---
            results = run_full_analysis(
                data_path=data_source, 
                battery_cap=BATTERY_CAP_KWH, 
                laadsnelheid=LAADSNELHEID_UREN, 
                rte=RTE, 
                verbruiker_type=CABINE_AANWEZIG,
                buy_offset_groot=BELPEX_BUY_PRICE_OFFSET_GROOT_EUR_KWH,
                buy_offset_klein=MEERKOST_KLEINVERBRUIKER_EUR_KWH,
                sell_offset=BELPEX_SELL_PRICE_OFFSET_EUR_KWH,
                peak_cost=COST_PEAK_EUR_PER_KW_MONTH
            )
            st.session_state.analysis_results = results
            # Reset de staat van de tabs en weekkeuze voor een nieuwe analyse
            if 'active_tab' in st.session_state:
                st.session_state.active_tab = "Gemiddelde Dag van de Maand"
            if 'selected_week' in st.session_state:
                del st.session_state.selected_week
    else:
        st.sidebar.error("Selecteer of upload eerst een bestand.")

# ==============================================================================
#                 WEERGAVE VAN RESULTATEN 
# ==============================================================================
if st.session_state.analysis_results is None:
    st.info("Selecteer een gegevensbestand, pas de parameters aan en klik op 'Analyse Uitvoeren'.")
else:
    try:
        df, DT_HOURS, unique_months, optimized_peaks = st.session_state.analysis_results
        
        # --- Navigatie met radio knoppen die de state onthouden ---
        tabs = ["Gemiddelde Dag van de Maand", "Wekelijks Overzicht", "Resultaten Samenvatting"]
        
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = tabs[0]

        active_tab = st.radio("Selecteer een overzicht:", tabs, key="active_tab", horizontal=True, label_visibility="collapsed")

        # --- Toon de content van de actieve tab ---
        if active_tab == "Gemiddelde Dag van de Maand":
            st.header("Gemiddelde Dag van de Maand")
            df_plot = df.copy()
            df_plot["Injection_neg"] = -df["Injection_KWh"].clip(lower=0.0)
            if "PV_KWh" not in df_plot.columns: df_plot["PV_KWh"] = 0
            
            fig1 = _monthly_avg_block(df_plot, ["Import_KWh","PV_KWh","Injection_neg"], ["Huidige afname (kWh)", "PV productie (kWh)", "Huidige injectie"], "Gemiddelde dag van de maand (zonder batterij)")
            if fig1: st.pyplot(fig1)

            df_plot2 = df.copy()
            df_plot2["Injection_after_neg"] = -df_plot2["Export_after_auto_kWh"].clip(lower=0.0)
            if "PV_KWh" not in df_plot2.columns: df_plot2["PV_KWh"] = 0
            fig2 = _monthly_avg_block(df_plot2, ["Grid_buy_after_auto_kWh","PV_KWh","Injection_after_neg"], ["Afname na batterij (kWh)", "PV productie (kWh)", "Injectie na batterij"], "Gemiddelde dag van de maand (met batterij autoconsumptie sturing)", soc_col="SoC_auto2_KWh", soc_divisor=20)
            if fig2: st.pyplot(fig2)

            df_plot3 = df.copy()
            df_plot3["Injection_da_neg"] = -df_plot3["Export_after_da_KWh"].clip(lower=0.0)
            if "PV_KWh" not in df_plot3.columns: df_plot3["PV_KWh"] = 0
            fig3 = _monthly_avg_block(df_plot3, ["Grid_total_da_KWh","PV_KWh","Injection_da_neg"], ["Afname na batterij (kWh)", "PV productie (kWh)", "Injectie na batterij"], "Gemiddelde dag van de maand (met batterij day-ahead sturing)", soc_col="SoC_da_KWh", soc_divisor=20)
            if fig3: st.pyplot(fig3)

            st.subheader("MAANDELIJKSE PIEKVERMOGENS (kW) VERGELIJKING")
            if 'Import_kW' not in df.columns: df['Import_kW'] = df['Import_KWh'] / DT_HOURS
            original_monthly_peak_kW = df.groupby('Maand_Jaar')['Import_kW'].max()
            peak_data = []
            for m in unique_months:
                original_peak = original_monthly_peak_kW.get(m, 0.0)
                optimized_peak = optimized_peaks.get(str(m), 0.0)
                reduction_kW = original_peak - optimized_peak
                monthly_peak_savings_eur = reduction_kW * COST_PEAK_EUR_PER_KW_MONTH
                peak_data.append({"Maand": str(m), "Origineel (kW)": original_peak, "Geoptimaliseerd (kW)": optimized_peak, "Reductie (kW)": reduction_kW, "Piekbesparing (€)": monthly_peak_savings_eur})
            peak_df = pd.DataFrame(peak_data)
            total_reduction_cost_savings = peak_df["Piekbesparing (€)"].sum()
            avg_row_data = {"Maand": "Gemiddelde / Totaal", "Origineel (kW)": peak_df["Origineel (kW)"].mean(), "Geoptimaliseerd (kW)": peak_df["Geoptimaliseerd (kW)"].mean(), "Reductie (kW)": peak_df["Reductie (kW)"].mean(), "Piekbesparing (€)": total_reduction_cost_savings}
            peak_df = pd.concat([peak_df, pd.DataFrame([avg_row_data])], ignore_index=True)
            st.dataframe(peak_df.style.format({"Origineel (kW)": "{:,.2f}", "Geoptimaliseerd (kW)": "{:,.2f}", "Reductie (kW)": "{:,.2f}", "Piekbesparing (€)": "{:,.2f}"}))

        elif st.session_state.active_tab == "Wekelijks Overzicht":
            st.header("Batterijsimulatie - Wekelijks Overzicht")

            # Data voorbereiding
            df["ISO_Week_Str"] = df["Date"].dt.strftime('%Y-W%U')
            week_keys = df["ISO_Week_Str"]
            price_buy = pd.to_numeric(df["Prijs_verbruik"], errors="coerce").fillna(0.0)
            price_sell = pd.to_numeric(df["Prijs_injectie"], errors="coerce").fillna(0.0)
            base_week  = (df["Import_KWh"]*price_buy - df["Injection_KWh"]*price_sell).groupby(week_keys).sum()
            da_week    = (df["Grid_total_da_KWh"]*price_buy - df["Export_after_da_KWh"]*price_sell).groupby(week_keys).sum()
            wk_index = sorted(base_week.index.unique())
            wk_df = pd.DataFrame({ "week": wk_index, "baseline": base_week.reindex(wk_index).fillna(0.0).values, "da": da_week.reindex(wk_index).fillna(0.0).values })
            wk_df["savings_da"] = wk_df["baseline"] - wk_df["da"]

            # State management
            if 'selected_week' not in st.session_state or st.session_state.selected_week not in wk_index:
                st.session_state.selected_week = wk_index[0] if wk_index else None

            st.subheader("Wekelijkse Besparing (klik om te selecteren)")
            
            # Altair Grafiek
            selection = alt.selection_point(fields=['week'], empty=True)
            color = alt.condition(selection, alt.value('orange'), alt.value('steelblue'))
            savings_chart = alt.Chart(wk_df).mark_bar().encode(
                x=alt.X('week:N', sort=None, title='Week'),
                y=alt.Y('savings_da:Q', title='Besparing (€)'),
                color=color,
                tooltip=['week', 'savings_da']
            ).add_params(selection).properties(height=300)
            
            # Gebruik 'param_1' zoals de foutmelding aangaf
            chart_selection = st.altair_chart(savings_chart, use_container_width=True, on_select="rerun", selection_mode=['param_1'])

            # Logica om de klik te verwerken en de app te herladen
            if (
                chart_selection
                and chart_selection.get("selection")
                and chart_selection["selection"].get("param_1")
            ):
                # De selectie is een lijst, we nemen het eerste element
                selected_data = chart_selection["selection"]["param_1"][0]
                selected_week_from_chart = selected_data["week"]

                if st.session_state.selected_week != selected_week_from_chart:
                    st.session_state.selected_week = selected_week_from_chart
                    st.rerun()
            
            # Knoppen en Selectbox
            col1, col2, col3 = st.columns([1, 4, 1])
            current_index = wk_index.index(st.session_state.selected_week)

            if col1.button("⬅️ Vorige Week"):
                st.session_state.selected_week = wk_index[max(0, current_index - 1)]
                st.rerun() 
            if col3.button("Volgende Week ➡️"):
                st.session_state.selected_week = wk_index[min(len(wk_index) - 1, current_index + 1)]
                st.rerun()
            
            col2.selectbox(
                "Geselecteerde week:",
                options=wk_index,
                key='selected_week',
                on_change=st.rerun 
            )

            # Detailgrafieken
            scenario = st.radio("Selecteer Scenario:", ('day-ahead', 'auto'), key='scenario_radio', horizontal=True)
            dd = df[week_keys == st.session_state.selected_week].copy().sort_values("Date")
            
            if not dd.empty:
                week_fig = make_week_series_fig(dd, st.session_state.selected_week, scenario, BATTERY_CAP_KWH)
                st.plotly_chart(week_fig, use_container_width=True)

        elif st.session_state.active_tab == "Resultaten Samenvatting":
            st.header("Resultaten Samenvatting")

            # --- Berekeningen voor tabellen (ongewijzigd) ---
            price_buy = df["Prijs_verbruik"]
            price_sell = df["Prijs_injectie"]
            baseline_cost = (df["Import_KWh"]*price_buy - df["Injection_KWh"]*price_sell).sum()
            auto_cost = (df["Grid_buy_after_auto_kWh"]*price_buy - df["Export_after_auto_kWh"]*price_sell).sum()

            dayahead_variable_energy_cost = (df["Grid_total_da_KWh"]*price_buy - df["Export_after_da_KWh"]*price_sell).sum()
            optimized_peak_cost = sum(COST_PEAK_EUR_PER_KW_MONTH * peak for peak in optimized_peaks.values())
            dayahead_total_cost = dayahead_variable_energy_cost + optimized_peak_cost
            
            if 'Import_kW' not in df.columns: df['Import_kW'] = df['Import_KWh'] / DT_HOURS
            original_monthly_peak_kW = df.groupby('Maand_Jaar')['Import_kW'].max()
            baseline_peak_cost = original_monthly_peak_kW.sum() * COST_PEAK_EUR_PER_KW_MONTH
            baseline_cost_with_peak = baseline_cost + baseline_peak_cost

            savings_variable_energy_dayahead = baseline_cost - dayahead_variable_energy_cost
            savings_peak_cost_dayahead = baseline_peak_cost - optimized_peak_cost
            total_savings_dayahead = baseline_cost_with_peak - dayahead_total_cost
            
            total_discharge_auto = df["Discharge_auto2_kWh"].sum()
            total_discharge_da = df["Discharge_da_kWh"].sum()
            
            total_charge_pv = df["Charge_from_PV_da_kWh"].sum()
            total_charge_cost_pv = (df["Charge_from_PV_da_kWh"] * df["Prijs_injectie"]).sum()
            avg_charge_pv = total_charge_cost_pv / total_charge_pv if total_charge_pv > 0 else 0.0
            total_discharge_load = df["Discharge_to_load_da_kWh"].sum()
            total_discharge_value_load = (df["Discharge_to_load_da_kWh"] * df["Prijs_verbruik"]).sum()
            avg_discharge_total = (total_discharge_value_load + (df["Discharge_to_grid_da_kWh"] * df["Prijs_injectie"]).sum()) / total_discharge_da if total_discharge_da > 0 else 0.0
            
            total_pv_kwh = df.get('PV_KWh', pd.Series([0])).sum()
            import_wb_kwh = df['Import_KWh'].sum()
            export_wb_kwh = df['Injection_KWh'].sum()
            avg_original_peak = original_monthly_peak_kW.mean() if not original_monthly_peak_kW.empty else 0
            export_auto_kwh = df['Export_after_auto_kWh'].sum()
            import_auto_kwh = df['Grid_buy_after_auto_kWh'].sum()
            avg_optimized_peak = np.mean(list(optimized_peaks.values())) if optimized_peaks else 0
            additional_discharge_da_kwh = total_discharge_da - total_discharge_auto
            avg_reduction_kw = avg_original_peak - avg_optimized_peak

            avg_discharge_delta_da = 0
            if additional_discharge_da_kwh > 0:
                numerator = (total_savings_dayahead - savings_peak_cost_dayahead - (avg_discharge_total - avg_charge_pv) * total_discharge_auto)
                avg_discharge_delta_da = numerator / additional_discharge_da_kwh
            
            # --- AANGEPAST: Tabel data met correcte opmaak en namen ---
            summary_table_data_ordered = [
                {'Parameter': 'Productie in kWh', 'Waarde': f"{total_pv_kwh:,.2f}".replace(',', '').replace('.', ','), 'Eenheid': 'kWh'},
                {'Parameter': 'Resterende net Afname in kWh (zonder batterij)', 'Waarde': f"{import_wb_kwh:,.2f}".replace(',', '').replace('.', ','), 'Eenheid': 'kWh'},
                {'Parameter': 'Injectie in kWh (zonder batterij)', 'Waarde': f"{export_wb_kwh:,.2f}".replace(',', '').replace('.', ','), 'Eenheid': 'kWh'},
                {'Parameter': 'Gemiddelde maandpiek (zonder batterij)', 'Waarde': f"{avg_original_peak:,.1f}".replace(',', '').replace('.', ','), 'Eenheid': 'kW'},
                {'Parameter': 'Batterij capaciteit in kwh', 'Waarde': f"{BATTERY_CAP_KWH:,.0f}", 'Eenheid': 'kWh'},
                {'Parameter': 'Injectie in kWh (na auto-cons)', 'Waarde': f"{export_auto_kwh:,.2f}".replace(',', '').replace('.', ','), 'Eenheid': 'kWh'},
                {'Parameter': 'Gemiddelde maandpiek (na optimalisatie)', 'Waarde': f"{avg_optimized_peak:,.1f}".replace(',', '').replace('.', ','), 'Eenheid': 'kW'},
                {'Parameter': 'Energie uit Batterij in kWh (day-ahead)', 'Waarde': f"{total_discharge_da:,.2f}".replace(',', '').replace('.', ','), 'Eenheid': 'kWh'},
                {'Parameter': 'Resterende net Afname in kWh (auto-cons)', 'Waarde': f"{import_auto_kwh:,.2f}".replace(',', '').replace('.', ','), 'Eenheid': 'kWh'},
                {'Parameter': 'Waarde zelfconsumptie (auto-cons)', 'Waarde': f"{(avg_discharge_total - avg_charge_pv):.4f}".replace(',', '').replace('.', ','), 'Eenheid': 'EUR/kWh'},
                {'Parameter': 'Waarde day-ahead sturing', 'Waarde': f"{avg_discharge_delta_da:.4f}".replace(',', '').replace('.', ','), 'Eenheid': 'EUR/kWh'},
                {'Parameter': 'kWh voor zelfconsumptie', 'Waarde': f"{total_discharge_auto:,.2f}".replace(',', '').replace('.', ','), 'Eenheid': 'kWh'},
                {'Parameter': 'kWh voor day-ahead sturing', 'Waarde': f"{additional_discharge_da_kwh:,.2f}".replace(',', '').replace('.', ','), 'Eenheid': 'kWh'},
                {'Parameter': 'Waarde peakshaving', 'Waarde': f"{COST_PEAK_EUR_PER_KW_MONTH*12:.2f}".replace(',', '').replace('.', ','), 'Eenheid': 'EUR/kW/jaar'},
                {'Parameter': 'Gemiddelde piek reductie', 'Waarde': f"{avg_reduction_kw:.4f}".replace(',', '').replace('.', ','), 'Eenheid': 'kW'},
                {'Parameter': 'Basiskost (variabel + piek)', 'Waarde': f"{baseline_cost_with_peak:,.2f}".replace(',', '').replace('.', ','), 'Eenheid': 'EUR'},
                {'Parameter': 'Besparing variabele energie (day-ahead)', 'Waarde': f"{savings_variable_energy_dayahead:,.2f}".replace(',', '').replace('.', ','), 'Eenheid': 'EUR'},
                {'Parameter': 'Besparing piekvermogen (day-ahead)', 'Waarde': f"{savings_peak_cost_dayahead:,.2f}".replace(',', '').replace('.', ','), 'Eenheid': 'EUR'},
            ]
            results_df_ordered = pd.DataFrame(summary_table_data_ordered).set_index('Parameter')
            
            st.subheader("SAMENVATTING RESULTATEN")
            st.dataframe(results_df_ordered)

            # --- Andere overzichten (deze blijven ongewijzigd) ---
            st.subheader("Kostentotalen")
            st.metric("Baseline Kosten", f"€ {baseline_cost_with_peak:,.2f}")
            st.metric("Kosten na Day-ahead", f"€ {dayahead_total_cost:,.2f}")

            st.subheader("Besparingen tov Baseline")
            st.metric("Totale besparing Day-ahead en peak shaving", f"€ {total_savings_dayahead:,.2f}")
            st.text(f"  ↳ Besp. variabele energie: € {savings_variable_energy_dayahead:,.2f}")
            st.text(f"  ↳ Besp. piekvermogen: € {savings_peak_cost_dayahead:,.2f}")
            
            st.subheader("Gemiddelde Prijzen voor Batterijstromen (€/kWh)")
            total_charge_cost_grid = (df["Charge_from_grid_da_kWh"] * df["Prijs_verbruik"]).sum()
            avg_charge_grid = total_charge_cost_grid / df["Charge_from_grid_da_kWh"].sum() if df["Charge_from_grid_da_kWh"].sum() > 0 else 0.0
            avg_charge_total = (total_charge_cost_pv + total_charge_cost_grid) / (total_charge_pv + df["Charge_from_grid_da_kWh"].sum()) if (total_charge_pv + df["Charge_from_grid_da_kWh"].sum()) > 0 else 0.0
            
            total_discharge_value_grid = (df["Discharge_to_grid_da_kWh"] * df["Prijs_injectie"]).sum()
            avg_discharge_grid = total_discharge_value_grid / df["Discharge_to_grid_da_kWh"].sum() if df["Discharge_to_grid_da_kWh"].sum() > 0 else 0.0
            avg_discharge_load = total_discharge_value_load / total_discharge_load if total_discharge_load > 0 else 0.0

            col1, col2 = st.columns(2)
            with col1:
                st.text("Laden")
                st.text(f"  Vanaf PV:       € {avg_charge_pv:.4f}")
                st.text(f"  Vanaf net:      € {avg_charge_grid:.4f}")
                st.text(f"  Totaal:             € {avg_charge_total:.4f}")
            with col2:
                st.text("Ontladen")
                st.text(f"  Naar verbruik: € {avg_discharge_load:.4f}")
                st.text(f"  Naar net:     € {avg_discharge_grid:.4f}")
                st.text(f"  Totaal:             € {avg_discharge_total:.4f}")

    except Exception as e:
        st.error(f"Er is een onverwachte fout opgetreden:")
        st.exception(e)