import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from typing import Optional
import pulp
import plotly.graph_objects as go
import plotly.subplots as make_subplots
import io

# --- Pagina Configuratie ---
st.set_page_config(layout="wide")

# =====================================================================================
# 1. HELPER FUNCTIES (grotendeels uit uw notebook)
# =====================================================================================

def _rename_aliases(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    lowmap = {str(c).strip().lower(): c for c in df_out.columns}

    def pick(*aliases, required=False):
        for a in aliases:
            if a in lowmap:
                return lowmap[a]
        if required:
            raise KeyError(f"Missing any of columns: {aliases}")
        return None

    col_date = pick("date", "datetime", "timestamp", required=True)
    col_import = pick("import_kwh", "import", "grid import (kwh)", "grid_import_kwh", required=True)
    col_inject = pick("injection_kwh", "injection", "export_kwh", "export (kwh)")
    col_pv = pick("pv_kwh", "zonne-opbrengst (kwh)", "pv", "solar_kwh")
    col_cons = pick("consumption_kwh", "verbruik (kwh)", "consumption")
    col_belpex = pick("belpex", required=True)

    rename = {
        col_date: "Date",
        col_import: "Import_KWh",
        col_belpex: "BELPEX",
    }
    if col_inject: rename[col_inject] = "Injection_KWh"
    if col_pv: rename[col_pv] = "PV_KWh"
    if col_cons: rename[col_cons] = "Consumption_KWh"

    df_out = df_out.rename(columns=rename)
    return df_out

def _monthly_avg_block(df, y_cols, labels, title, soc_col=None, soc_label="Battery SoC (kWh)", soc_divisor=1):
    dfx = df.copy()
    dfx["Date"] = pd.to_datetime(dfx["Date"]).dt.round('15min')
    dfx["Month"] = dfx["Date"].dt.to_period("M")
    months = sorted(dfx["Month"].unique())
    
    if not months:
        return None

    x_all, series_all, soc_all = [], [[] for _ in y_cols], []
    xticks, xlabels, seps, pos = [], [], [], 0

    for m in months:
        sub = dfx[dfx["Month"] == m].copy()
        sub["H"] = sub["Date"].dt.hour
        sub["M"] = sub["Date"].dt.minute
        grp = sub.groupby(["H", "M"]).mean(numeric_only=True)
        idx = sorted(grp.index, key=lambda z: (z[0], z[1]))
        prof = grp.loc[idx]
        
        steps = len(prof)
        x_seg = np.arange(steps) + pos
        x_all.append(x_seg)
        for j, col in enumerate(y_cols):
            series_all[j].append(prof.get(col, pd.Series(np.zeros(steps))).to_numpy())
        if soc_col:
            soc_all.append((prof.get(soc_col, pd.Series(np.zeros(steps))).to_numpy()) / soc_divisor)
        
        seps.append(pos)
        xticks.append(pos + steps / 2)
        xlabels.append(m.to_timestamp().strftime("%b %Y"))
        pos += steps

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.concatenate(x_all)
    colors = ["tab:blue", "tab:red", "black", "tab:green", "tab:orange", "tab:cyan"]

    for j, (lbl, col) in enumerate(zip(labels, y_cols)):
        y = np.concatenate(series_all[j])
        ax.plot(x, y, label=lbl, color=colors[j % len(colors)], linewidth=1.8)
    
    if soc_col:
        soc = np.concatenate(soc_all)
        soc_plot_color = "tab:purple"
        label = f"{soc_label} / {soc_divisor}" if soc_divisor != 1 else soc_label
        ax.plot(x, soc, label=label, color=soc_plot_color, linestyle="--", linewidth=1.6)

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

# =====================================================================================
# 2. GECACHETE ANALYSE FUNCTIE
# =====================================================================================

@st.cache_data
def run_full_analysis(uploaded_file_content, battery_cap, start_soc, reserve_soc, rte, charge_hours, discharge_hours, buy_offset, sell_offset, peak_cost):
    """
    Deze functie voert de volledige analyse uit en wordt gecachet om herhaling te voorkomen.
    """
    df = pd.read_excel(io.BytesIO(uploaded_file_content))
    
    # --- Data Inladen & Voorbereiden ---
    df = _rename_aliases(df)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M")
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.drop_duplicates(subset=['Date'], keep='first')
    
    if "Injection_KWh" not in df.columns:
        if {"PV_KWh", "Consumption_KWh"}.issubset(df.columns):
            df["Injection_KWh"] = (df["PV_KWh"] - df["Consumption_KWh"]).clip(lower=0.0)
        else:
            df["Injection_KWh"] = 0.0

    belpex_price_eur_kwh = pd.to_numeric(df["BELPEX"], errors="coerce").fillna(0.0) / 1000.0
    df["Prijs_verbruik"] = buy_offset + belpex_price_eur_kwh
    df["Prijs_injectie"] = sell_offset + belpex_price_eur_kwh

    if len(df) > 1:
        dt_s = df["Date"].diff().dropna().dt.total_seconds().mode()
        dt_hours = float(dt_s.iloc[0] / 3600.0) if not dt_s.empty else 1.0
    else:
        dt_hours = 1.0

    charge_cap_per_hour = battery_cap / max(charge_hours, 1e-9)
    discharge_cap_per_hour = battery_cap / max(discharge_hours, 1e-9)
    charge_cap_step = charge_cap_per_hour * dt_hours
    discharge_cap_step = discharge_cap_per_hour * dt_hours
    
    price_buy = pd.to_numeric(df["Prijs_verbruik"], errors="coerce").fillna(0.0)
    price_sell = pd.to_numeric(df["Prijs_injectie"], errors="coerce").fillna(0.0)
    imp0 = pd.to_numeric(df["Import_KWh"], errors="coerce").fillna(0.0)
    inj0 = pd.to_numeric(df["Injection_KWh"], errors="coerce").fillna(0.0)
    
    baseline_cost = float((imp0 * price_buy - inj0 * price_sell).sum())
    
    # --- Auto-consumptie Simulatie ---
    N = len(df)
    soc = start_soc
    chg_pv_auto, dis_auto, grid_after, export_after, soc_auto = [np.zeros(N) for _ in range(5)]

    for t in range(N):
        headroom_out = battery_cap - soc
        take = min(inj0[t], charge_cap_step / max(rte, 1e-9), headroom_out / max(rte, 1e-9))
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
    
    auto_cost = float((df["Grid_buy_after_auto_kWh"] * price_buy - df["Export_after_auto_kWh"] * price_sell).sum())

    # --- MILP Optimalisatie ---
    import_arr = imp0.to_numpy(float)
    df['Maand_Jaar'] = df['Date'].dt.to_period('M')
    month_of_t = [df['Maand_Jaar'].iloc[t] for t in range(N)]
    unique_months = sorted(df['Maand_Jaar'].unique())
    
    prob = pulp.LpProblem("battery_profit_max", pulp.LpMaximize)

    charge_g = pulp.LpVariable.dicts("charge_grid", range(N), lowBound=0)
    charge_p = pulp.LpVariable.dicts("charge_pv", range(N), lowBound=0)
    discharge_g = pulp.LpVariable.dicts("discharge_grid", range(N), lowBound=0)
    discharge_l = pulp.LpVariable.dicts("discharge_load", range(N), lowBound=0)
    soc_t = pulp.LpVariable.dicts("soc", range(N + 1), lowBound=0, upBound=battery_cap)
    pv_to_grid = pulp.LpVariable.dicts("pv_to_grid", range(N), lowBound=0)
    grid_to_load = pulp.LpVariable.dicts("grid_to_load", range(N), lowBound=0)
    peak_power_month_kw = pulp.LpVariable.dicts("peak_power_month_kw", unique_months, lowBound=0)
    grid_total_power_kw = pulp.LpVariable.dicts("grid_total_power_kw", range(N), lowBound=0)
    
    prob += (pulp.lpSum((price_sell[t] * (discharge_g[t] + pv_to_grid[t])) - (price_buy[t] * (grid_to_load[t] + charge_g[t])) for t in range(N))
             - pulp.lpSum(peak_cost * peak_power_month_kw[m] for m in unique_months))
    
    for t in range(N):
        prob += soc_t[t+1] == soc_t[t] + rte * (charge_g[t] + charge_p[t]) - (discharge_g[t] + discharge_l[t])
        prob += charge_g[t] + charge_p[t] <= charge_cap_step
        prob += discharge_g[t] + discharge_l[t] <= discharge_cap_step
        prob += soc_t[t] >= reserve_soc
        prob += charge_p[t] + pv_to_grid[t] == inj0.iloc[t] if "Injection_KWh" in df.columns else 0
        prob += grid_total_power_kw[t] == (grid_to_load[t] + charge_g[t]) / dt_hours
        prob += peak_power_month_kw[month_of_t[t]] >= grid_total_power_kw[t]
        prob += grid_to_load[t] + discharge_l[t] == import_arr[t]
    
    prob += soc_t[0] == start_soc
    prob += soc_t[N] == start_soc
    
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # --- Resultaten Verzamelen ---
    df["Charge_from_grid_da_kWh"] = [pulp.value(charge_g[t]) for t in range(N)]
    df["Charge_from_PV_da_kWh"] = [pulp.value(charge_p[t]) for t in range(N)]
    df["Discharge_da_kWh"] = [pulp.value(discharge_g[t]) + pulp.value(discharge_l[t]) for t in range(N)]
    df["SoC_da_KWh"] = [pulp.value(soc_t[t+1]) for t in range(N)]
    df["Grid_total_da_KWh"] = [pulp.value(grid_to_load[t]) + pulp.value(charge_g[t]) for t in range(N)]
    df["Export_after_da_KWh"] = [pulp.value(discharge_g[t]) + pulp.value(pv_to_grid[t]) for t in range(N)]
    df["Discharge_to_load_da_kWh"] = [pulp.value(discharge_l[t]) for t in range(N)]
    df["Discharge_to_grid_da_kWh"] = [pulp.value(discharge_g[t]) for t in range(N)]

    df['Import_kW'] = df['Import_KWh'] / dt_hours
    original_monthly_peak_kW = df.groupby('Maand_Jaar')['Import_kW'].max()

    return df, original_monthly_peak_kW, peak_power_month_kw, unique_months

# =====================================================================================
# 3. STREAMLIT UI
# =====================================================================================

st.title("🔋 Batterijsimulatie voor Peak Shaving & Energie-arbitrage")

# --- Zijbalk met configuratie-opties ---
st.sidebar.header("Configuratie")
uploaded_file = st.sidebar.file_uploader("Upload uw Excel-bestand", type=["xlsx", "xls"])

st.sidebar.subheader("Batterijparameters")
BATTERY_CAP_KWH = st.sidebar.number_input("Batterijcapaciteit (kWh)", value=450.0, step=10.0)
CHARGE_HOURS_TO_FULL = st.sidebar.number_input("Laadtijd tot vol (uren)", value=2.0, step=0.25)
DISCHARGE_HOURS_TO_EMPTY = st.sidebar.number_input("Ontlaadtijd tot leeg (uren)", value=2.0, step=0.25)
RTE = st.sidebar.slider("Round-Trip Efficiency (RTE)", 0.0, 1.0, 0.88, 0.01)
START_SOC_KWH = st.sidebar.number_input("Start SoC (kWh)", value=0.0)
RESERVE_SOC_KWH = st.sidebar.number_input("Reserve SoC (kWh)", value=0.0)

st.sidebar.subheader("Kostenparameters")
BELPEX_BUY_PRICE_OFFSET_EUR_KWH = st.sidebar.number_input("Prijs-offset Verbruik (€/kWh)", value=0.029, format="%.4f")
BELPEX_SELL_PRICE_OFFSET_EUR_KWH = st.sidebar.number_input("Prijs-offset Injectie (€/kWh)", value=-0.020, format="%.4f")
COST_PEAK_EUR_PER_KW_MONTH = st.sidebar.number_input("Piekkosten (€/kW/maand)", value=5.0, step=0.5)

# --- Hoofdpagina ---
if uploaded_file is not None:
    file_content = uploaded_file.getvalue()
    
    # Voer de volledige analyse uit (gebruikt cache indien mogelijk)
    df, original_peaks, optimized_peaks, unique_months = run_full_analysis(
        file_content, BATTERY_CAP_KWH, START_SOC_KWH, RESERVE_SOC_KWH, RTE, 
        CHARGE_HOURS_TO_FULL, DISCHARGE_HOURS_TO_EMPTY, 
        BELPEX_BUY_PRICE_OFFSET_EUR_KWH, BELPEX_SELL_PRICE_OFFSET_EUR_KWH,
        COST_PEAK_EUR_PER_KW_MONTH
    )

    # --- Bereken de samenvattende statistieken ---
    price_buy = df["Prijs_verbruik"]
    price_sell = df["Prijs_injectie"]

    baseline_cost = (df["Import_KWh"] * price_buy - df["Injection_KWh"] * price_sell).sum()
    auto_cost = (df["Grid_buy_after_auto_kWh"] * price_buy - df["Export_after_auto_kWh"] * price_sell).sum()
    dayahead_variable_cost = (df["Grid_total_da_KWh"] * price_buy - df["Export_after_da_KWh"] * price_sell).sum()

    baseline_peak_cost = sum(COST_PEAK_EUR_PER_KW_MONTH * peak for peak in original_peaks)
    optimized_peak_cost = sum(COST_PEAK_EUR_PER_KW_MONTH * pulp.value(optimized_peaks[m]) for m in unique_months)

    baseline_total_cost = baseline_cost + baseline_peak_cost
    auto_total_cost = auto_cost + baseline_peak_cost
    dayahead_total_cost = dayahead_variable_cost + optimized_peak_cost

    # Creëer tabbladen voor overzicht
    tab1, tab2 = st.tabs(["📊 Samenvatting & Maandprofielen", "🗓️ Interactieve Wekelijkse Analyse"])

    with tab1:
        st.header("Analyse Samenvatting")
        
        # --- Maandelijkse Piekvermogens ---
        st.subheader("Maandelijkse Piekvermogens (kW)")
        peak_data = []
        for m in unique_months:
            original = original_peaks.get(m, 0.0)
            optimized = pulp.value(optimized_peaks[m])
            reduction = original - optimized
            savings = reduction * COST_PEAK_EUR_PER_KW_MONTH
            peak_data.append([str(m), original, optimized, reduction, savings])
        
        peak_df = pd.DataFrame(peak_data, columns=["Maand", "Origineel (kW)", "Geoptimaliseerd (kW)", "Reductie (kW)", "Piekbesparing (€)"])
        st.dataframe(peak_df.style.format("{:.2f}", subset=peak_df.columns[1:]))

        # --- Kosten en Besparingen ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Kostentotalen (incl. Piek)")
            cost_data = {
                "Scenario": ["Baseline", "Auto-consumptie", "Day-ahead (MILP)"],
                "Totale Kosten (€)": [baseline_total_cost, auto_total_cost, dayahead_total_cost]
            }
            cost_df = pd.DataFrame(cost_data)
            st.dataframe(cost_df.style.format({"Totale Kosten (€)": "€{:,.2f}"}))
        
        with col2:
            st.subheader("Besparingen vs. Baseline")
            savings_data = {
                "Scenario": ["Auto-consumptie", "Day-ahead (MILP)"],
                "Totale Besparing (€)": [baseline_total_cost - auto_total_cost, baseline_total_cost - dayahead_total_cost],
                "Besparing Variabel (€)": [baseline_cost - auto_cost, baseline_cost - dayahead_variable_cost],
                "Besparing Piek (€)": [0, baseline_peak_cost - optimized_peak_cost]
            }
            savings_df = pd.DataFrame(savings_data)
            st.dataframe(savings_df.style.format(formatter={col: "€{:,.2f}" for col in savings_df.columns[1:]}))

        # --- Maandelijkse Plots ---
        st.header("Gemiddelde Dagprofielen per Maand")
        
        df_plot = df.copy()
        df_plot["Injection_neg"] = -df["Injection_KWh"].clip(lower=0.0)
        fig_baseline = _monthly_avg_block(df_plot, ["Import_KWh", "PV_KWh", "Injection_neg"], ["Huidige afname", "PV productie", "Huidige injectie"], "Zonder Batterij")
        st.pyplot(fig_baseline)
        
        df_plot2 = df.copy()
        df_plot2["Injection_after_neg"] = -df_plot2["Export_after_auto_kWh"].clip(lower=0.0)
        fig_auto = _monthly_avg_block(df_plot2, ["Grid_buy_after_auto_kWh", "PV_KWh", "Injection_after_neg"], ["Afname na batterij", "PV productie", "Injectie na batterij"], "Met Batterij (Auto-consumptie)", "SoC_auto2_KWh", soc_divisor=5)
        st.pyplot(fig_auto)
        
        df_plot3 = df.copy()
        df_plot3["Injection_da_neg"] = -df_plot3["Export_after_da_KWh"].clip(lower=0.0)
        fig_da = _monthly_avg_block(df_plot3, ["Grid_total_da_KWh", "PV_KWh", "Injection_da_neg"], ["Afname na batterij", "PV productie", "Injectie na batterij"], "Met Batterij (Day-ahead sturing)", "SoC_da_KWh", soc_divisor=5)
        st.pyplot(fig_da)
        
    with tab2:
        st.header("Wekelijkse Analyse")
        
        # --- Wekelijkse Dataframes Voorbereiden ---
        df["ISO_Week"] = df["Date"].dt.isocalendar().week.astype(int)
        df["ISO_Year"] = df["Date"].dt.isocalendar().year.astype(int)
        week_keys = df["ISO_Year"].astype(str) + "-W" + df["ISO_Week"].astype(str).str.zfill(2)
        
        wk_index = sorted(week_keys.unique())
        
        # --- Interactieve Selectie ---
        selected_week = st.selectbox("Kies een week voor detailweergave:", options=wk_index)
        
        # --- Plotly Figuren (Recreatie van Dash) ---
        base_week  = (df["Import_KWh"]*price_buy - df["Injection_KWh"]*price_sell).groupby(week_keys).sum()
        auto_week  = (df["Grid_buy_after_auto_kWh"]*price_buy - df["Export_after_auto_kWh"]*price_sell).groupby(week_keys).sum()
        da_week    = (df["Grid_total_da_KWh"]*price_buy - df["Export_after_da_KWh"]*price_sell).groupby(week_keys).sum()

        wk_df = pd.DataFrame({
            "week": wk_index,
            "baseline": base_week.reindex(wk_index, fill_value=0),
            "auto": auto_week.reindex(wk_index, fill_value=0),
            "da": da_week.reindex(wk_index, fill_value=0),
        })
        wk_df["savings_da"] = wk_df["baseline"] - wk_df["da"]
        
        # Figuur voor wekelijkse besparingen
        fig_savings_picker = go.Figure()
        fig_savings_picker.add_trace(go.Bar(x=wk_df["week"], y=wk_df["savings_da"], name="Besparing Day-ahead (€)", opacity=0.7))
        fig_savings_picker.add_vline(x=selected_week, line_width=2, line_dash="dash", line_color="red")
        fig_savings_picker.update_layout(title="Wekelijkse Besparing", xaxis_title="ISO Week", yaxis_title="€")
        st.plotly_chart(fig_savings_picker, use_container_width=True)

        # Figuur voor wekelijkse stromen
        week_dd = df[week_keys == selected_week].copy()
        if not week_dd.empty:
            scenario_choice = st.radio("Kies scenario voor de detailgrafiek:", ["Day-ahead", "Auto-consumptie"], horizontal=True)
            scenario_map = {"Day-ahead": "da", "Auto-consumptie": "auto"}
            
            # Functie om week-detail plot te maken (aangepast uit uw notebook)
            def make_week_series_fig(dd, scenario="da"):
                if scenario == "auto":
                    y_import_after, y_export_after, soc, title = dd["Grid_buy_after_auto_kWh"], -dd["Export_after_auto_kWh"], dd.get("SoC_auto2_KWh"), f"Week {selected_week} (Auto-cons)"
                else:
                    y_import_after, y_export_after, soc, title = dd["Grid_total_da_KWh"], -dd["Export_after_da_KWh"], dd.get("SoC_da_KWh"), f"Week {selected_week} (Day-ahead)"

                fig = make_subplots.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15, subplot_titles=("Energiestromen", "Prijzen en SoC"), specs=[[{"secondary_y": False}], [{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=dd["Date"], y=dd.get("PV_KWh", 0), name="PV (kWh)", marker_color='gold'), row=1, col=1)
                fig.add_trace(go.Bar(x=dd["Date"], y=dd["Import_KWh"], name="Import (baseline)", marker_color='rgba(255,0,0,0.5)', opacity=0.5), row=1, col=1)
                fig.add_trace(go.Bar(x=dd["Date"], y=-dd["Injection_KWh"], name="Injectie (baseline)", marker_color='rgba(0,128,0,0.5)', opacity=0.5), row=1, col=1)
                fig.add_trace(go.Scatter(x=dd["Date"], y=y_import_after, mode="lines", name=f"Import ({scenario})", line=dict(color='red', width=3)), row=1, col=1)
                fig.add_trace(go.Scatter(x=dd["Date"], y=y_export_after, mode="lines", name=f"Injectie ({scenario})", line=dict(color='green', width=3)), row=1, col=1)
                fig.add_trace(go.Scatter(x=dd["Date"], y=dd["Prijs_verbruik"], mode="lines", name="Prijs afname (€/kWh)", line=dict(color='red')), row=2, col=1)
                fig.add_trace(go.Scatter(x=dd["Date"], y=dd["Prijs_injectie"], mode="lines", name="Prijs injectie (€/kWh)", line=dict(color='green')), row=2, col=1)
                fig.add_trace(go.Scatter(x=dd["Date"], y=soc, mode="lines", name="Batterij SoC (kWh)", line=dict(dash="dash", color='purple')), row=2, col=1, secondary_y=True)
                fig.update_layout(title=title, hovermode="x unified", height=600, legend=dict(orientation="h", y=-0.15))
                fig.update_yaxes(title_text="kWh", row=1, col=1)
                fig.update_yaxes(title_text="€/kWh", row=2, col=1, secondary_y=False)
                fig.update_yaxes(title_text="SoC (kWh)", row=2, col=1, secondary_y=True)
                return fig
            
            st.plotly_chart(make_week_series_fig(week_dd, scenario=scenario_map[scenario_choice]), use_container_width=True)

else:
    st.info("Upload een Excel-bestand om de analyse te starten.")