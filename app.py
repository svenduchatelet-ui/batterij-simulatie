import streamlit as st
import numpy as np
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

# --- Configuratie ---
# In een echte app zou de data HIER worden geladen of gesimuleerd.
# Voor deze conversie gebruiken we mock data en variabelen uit de notebook.

# Mock Data (Simuleert de input data van de notebook)
@st.cache_data
def load_mock_data():
    """Genereert mock data om de notebook-structuur te simuleren."""
    periods = 96  # 15-minuten intervallen per dag
    np.random.seed(42) # Voor reproduceerbaarheid

    # Maak een tijdindex voor één dag
    time_index = pd.date_range(start='2023-01-01', periods=periods, freq='15min')

    # Basis Verbruik (met wat variatie)
    base_consumption = 200 + 50 * np.sin(np.linspace(0, 2 * np.pi, periods))
    consumption_noise = np.random.normal(0, 30, periods)
    total_consumption = (base_consumption + consumption_noise).clip(min=50)

    # Basis Injectie (met wat zonne-energie simulatie overdag)
    base_injection = np.maximum(0, 100 * np.sin(np.linspace(0, np.pi, periods)) - 30)
    injection_noise = np.random.normal(0, 5, periods)
    total_injection = np.maximum(0, base_injection + injection_noise)

    # Prijsdata (met piek in de ochtend en avond)
    prices = 0.15 + 0.10 * np.sin(np.linspace(0, 4 * np.pi, periods))
    prices = np.maximum(-0.05, prices + np.random.normal(0, 0.05, periods))
    price_consumption = prices + 0.05
    price_injection = prices - 0.05

    df = pd.DataFrame(
        {
            'Vraag': total_consumption,
            'Aanbod': total_injection,
            'Prijs_verbruik': price_consumption,
            'Prijs_injectie': price_injection,
        },
        index=time_index
    )
    # Nettovraag is Vraag - Aanbod. Positief is kopen, negatief is verkopen.
    df['Nettovraag_Baseline'] = df['Vraag'] - df['Aanbod']
    df['Time_Steps'] = range(periods)
    df.index.name = 'Tijd'

    return df

# Optimalisatiefunctie gebaseerd op de logica uit de notebook
def optimize_battery(df: pd.DataFrame, cap_kwh: float, max_p_kw: float, soh: float, step_h: float, penalty_peak: float, verbose: bool = False) -> tuple[pd.DataFrame, float, float]:
    """
    Voert de Day-Ahead Peak Shaving optimalisatie uit met PuLP.

    Args:
        df: DataFrame met Nettovraag_Baseline en prijsinformatie.
        cap_kwh: Batterijcapaciteit in kWh.
        max_p_kw: Maximaal laad/ontlaadvermogen in kW.
        soh: State of Health (efficiëntie).
        step_h: Tijdsinterval in uren (bijv. 0.25 voor 15 minuten).
        penalty_peak: De straf (kosten) voor elke kW piekvermogen.

    Returns:
        Een tuple: (Resultaat DataFrame, geoptimaliseerde kosten, baseline piek)
    """
    time_steps = df['Time_Steps'].tolist()
    consumption_prices = df['Prijs_verbruik'].tolist()
    injection_prices = df['Prijs_injectie'].tolist()
    baseline_net_demand = df['Nettovraag_Baseline'].tolist()

    # --- 1. Model en Variabelen ---
    prob = pulp.LpProblem("Battery_Optimization", pulp.LpMinimize)

    # Batterij laad/ontlaad (kW)
    P_charge = pulp.LpVariable.dicts("P_charge", time_steps, lowBound=0, upBound=max_p_kw)
    P_discharge = pulp.LpVariable.dicts("P_discharge", time_steps, lowBound=0, upBound=max_p_kw)

    # Batterij SOC (State of Charge, kWh)
    SOC = pulp.LpVariable.dicts("SOC", time_steps, lowBound=0, upBound=cap_kwh)

    # Netto Vraag na optimalisatie (kW)
    Net_demand_opt = pulp.LpVariable.dicts("Net_demand_opt", time_steps, cat='Continuous')

    # Piekvermogen (het te minimaliseren doel)
    P_peak = pulp.LpVariable("P_peak", lowBound=0)

    # --- 2. Doelfunctie (Kosten minimaliseren) ---
    # De kosten bestaan uit variabele energiekosten + piekvermogen kosten
    energy_cost = pulp.lpSum(
        (Net_demand_opt[t] * consumption_prices[t] * step_h)
        for t in time_steps
    )
    peak_cost = P_peak * penalty_peak

    prob += energy_cost + peak_cost, "Total_Cost"

    # --- 3. Beperkingen ---

    # A. SOC dynamiek
    for t in time_steps:
        if t == 0:
            # SOC bij t=0 (Standaard naar 0.5 * CAP)
            prob += SOC[t] == 0.5 * cap_kwh + (P_charge[t] * soh - P_discharge[t] / soh) * step_h
        else:
            # SOC update: vorige SOC + (laden - ontladen)
            prob += SOC[t] == SOC[t-1] + (P_charge[t] * soh - P_discharge[t] / soh) * step_h

    # B. Netto Vraag Balans
    # Nieuwe Netto Vraag = Baseline - Ontladen + Laden
    for t in time_steps:
        prob += Net_demand_opt[t] == baseline_net_demand[t] - P_discharge[t] + P_charge[t]

    # C. Piekvermogen Beperking
    # Het piekvermogen (P_peak) moet groter zijn dan elk moment van de geoptimaliseerde netto vraag.
    for t in time_steps:
        # Dit is de Piek Vraag (import van het net)
        prob += P_peak >= Net_demand_opt[t], f"Peak_Constraint_{t}"

    # D. Laad/Ontlaad uitsluiting (kan niet tegelijkertijd)
    # Dit wordt impliciet opgelost door de lineaire Doelfunctie,
    # maar we kunnen het toevoegen voor de zekerheid of als de efficiëntie > 1 zou zijn.
    # In PuLP is de uitsluiting van P_charge en P_discharge in de meeste energie-optimalisaties
    # niet nodig omdat het minimaliseren van de kosten ervoor zorgt dat ze niet tegelijkertijd lopen.

    # --- 4. Oplossen ---
    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=0)) # msg=0 onderdrukt de output van de solver
    except Exception as e:
        st.error(f"Fout tijdens optimalisatie: {e}")
        return None, 0, 0

    # --- 5. Resultaten Verzamelen ---
    if pulp.LpStatus[prob.status] != 'Optimal':
        st.warning(f"Solver Status: {pulp.LpStatus[prob.status]}. Geen optimale oplossing gevonden.")
        return None, 0, 0

    # Gegevens extraheren
    df_results = df.copy()
    df_results['P_charge_opt'] = [P_charge[t].varValue if P_charge[t].varValue is not None else 0 for t in time_steps]
    df_results['P_discharge_opt'] = [P_discharge[t].varValue if P_discharge[t].varValue is not None else 0 for t in time_steps]
    df_results['SOC_opt'] = [SOC[t].varValue if SOC[t].varValue is not None else 0 for t in time_steps]
    df_results['Nettovraag_Opt'] = [Net_demand_opt[t].varValue if Net_demand_opt[t].varValue is not None else 0 for t in time_steps]

    optimal_cost = pulp.value(prob.objective)

    # Bereken Baseline Kosten
    baseline_energy_cost = pulp.lpSum(
        (df['Nettovraag_Baseline'][t] * consumption_prices[t] * step_h)
        for t in time_steps
        if df['Nettovraag_Baseline'][t] > 0
    )
    # Baseline piek is de maximale import (positieve Nettovraag_Baseline)
    baseline_peak = df['Nettovraag_Baseline'].clip(lower=0).max()
    baseline_cost_with_peak = pulp.value(baseline_energy_cost) + baseline_peak * penalty_peak


    return df_results, optimal_cost, baseline_peak

# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="Day-Ahead Peak Shaving Optimalisatie")
st.title("🔋 Day-Ahead Peak Shaving Optimalisatie (met PuLP)")

# Sidebar voor configuratie
st.sidebar.header("⚙️ Batterij & Kosten Parameters")

# Sidebar - Batterij Parameters
BATTERY_CAP_KWH = st.sidebar.slider("Batterij Capaciteit (kWh)", min_value=50.0, max_value=500.0, value=150.0, step=10.0)
MAX_P_KW = st.sidebar.slider("Max. Vermogen (kW) (laden/ontladen)", min_value=10.0, max_value=200.0, value=75.0, step=5.0)
SOH = st.sidebar.slider("State of Health (Efficiëntie)", min_value=0.85, max_value=1.0, value=0.95, step=0.01)

# Sidebar - Kosten Parameters
PEAK_PENALTY = st.sidebar.number_input("Piekvermogen Kosten (€/kW)", min_value=5.0, max_value=100.0, value=40.0, step=1.0)
TIME_STEP_H = 0.25 # 15 minuten intervallen

st.sidebar.markdown(f"""
---
#### Simulatie Details
* **Tijdsinterval:** 15 minuten ({TIME_STEP_H} uur)
* **Optimalisatiemodel:** Lineaire Programmering (PuLP)
""")

# Laad de mock data
df_data = load_mock_data()

# Voer de optimalisatie uit
df_results, optimal_cost, baseline_peak = optimize_battery(
    df_data,
    cap_kwh=BATTERY_CAP_KWH,
    max_p_kw=MAX_P_KW,
    soh=SOH,
    step_h=TIME_STEP_H,
    penalty_peak=PEAK_PENALTY
)

if df_results is not None:
    # Berekeningen voor de samenvattingstabel
    optimal_peak = df_results['Nettovraag_Opt'].clip(lower=0).max()
    peak_reduction = baseline_peak - optimal_peak

    # Baseline kosten zonder piek
    baseline_energy_cost = df_data.apply(
        lambda row: row['Nettovraag_Baseline'] * row['Prijs_verbruik'] * TIME_STEP_H
        if row['Nettovraag_Baseline'] > 0 else 0,
        axis=1
    ).sum()
    baseline_cost_with_peak = baseline_energy_cost + baseline_peak * PEAK_PENALTY

    # Geoptimaliseerde kosten zonder piek
    optimal_energy_cost = df_results.apply(
        lambda row: row['Nettovraag_Opt'] * row['Prijs_verbruik'] * TIME_STEP_H
        if row['Nettovraag_Opt'] > 0 else 0,
        axis=1
    ).sum()
    optimal_cost_with_peak = optimal_energy_cost + optimal_peak * PEAK_PENALTY

    # Besparingen
    savings_total = baseline_cost_with_peak - optimal_cost_with_peak
    savings_variable_energy = baseline_energy_cost - optimal_energy_cost
    savings_peak_cost = baseline_peak * PEAK_PENALTY - optimal_peak * PEAK_PENALTY
    
    # Gemiddelde reductie (indien > 0)
    if peak_reduction > 0:
        avg_reduction_kw = peak_reduction
    else:
        avg_reduction_kw = 0

    # --- Resultaten Weergeven ---
    st.header("Overzicht Resultaten")

    col1, col2, col3 = st.columns(3)

    # 1. Totaal Kosten
    col1.metric(
        label="Totale Besparing",
        value=f"€ {savings_total:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
        delta=f"Basiskost: € {baseline_cost_with_peak:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    )

    # 2. Piekreductie
    col2.metric(
        label="Piekvermogen Reductie",
        value=f"{peak_reduction:,.2f} kW".replace(",", "X").replace(".", ",").replace("X", "."),
        delta=f"Van {baseline_peak:,.2f} kW naar {optimal_peak:,.2f} kW".replace(",", "X").replace(".", ",").replace("X", ".")
    )

    # 3. Kostenverdeling
    col3.metric(
        label="Besparing Piekvermogen",
        value=f"€ {savings_peak_cost:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
        delta=f"Besparing Variabele Energie: € {savings_variable_energy:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    )

    st.subheader("Gedetailleerde Samenvatting")

    summary_table_data_ordered = [
        {'Parameter': 'Totale Basiskost', 'Waarde': f"{baseline_cost_with_peak:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."), 'Eenheid': 'EUR'},
        {'Parameter': 'Totale Geoptimaliseerde Kost', 'Waarde': f"{optimal_cost_with_peak:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."), 'Eenheid': 'EUR'},
        {'Parameter': 'Totale Besparing', 'Waarde': f"{savings_total:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."), 'Eenheid': 'EUR'},
        {'Parameter': 'Basispiek', 'Waarde': f"{baseline_peak:,.4f}".replace(",", "X").replace(".", ",").replace("X", "."), 'Eenheid': 'kW'},
        {'Parameter': 'Geoptimaliseerde Piek', 'Waarde': f"{optimal_peak:,.4f}".replace(",", "X").replace(".", ",").replace("X", "."), 'Eenheid': 'kW'},
        {'Parameter': 'Reductie Piek', 'Waarde': f"{peak_reduction:,.4f}".replace(",", "X").replace(".", ",").replace("X", "."), 'Eenheid': 'kW'},
    ]
    st.dataframe(pd.DataFrame(summary_table_data_ordered).set_index('Parameter'))


    # --- Plotly Grafieken ---
    st.header("Visualisatie van Resultaten")

    tab1, tab2, tab3 = st.tabs(["Nettovraag & Peak Shaving", "Batterij SOC & Prijs", "Nettovraag & Prijzen"])

    # 1. Nettovraag Plot (Hoofdplot van de optimalisatie)
    with tab1:
        fig1 = go.Figure()
        
        # Baseline
        fig1.add_trace(go.Scatter(
            x=df_results.index, 
            y=df_results['Nettovraag_Baseline'], 
            mode='lines', 
            name='Baseline Nettovraag (kW)', 
            line=dict(color='gray', dash='dash')
        ))
        
        # Geoptimaliseerd
        fig1.add_trace(go.Scatter(
            x=df_results.index, 
            y=df_results['Nettovraag_Opt'], 
            mode='lines', 
            name='Geoptimaliseerde Nettovraag (kW)', 
            line=dict(color='darkgreen', width=2)
        ))
        
        # Pieklijn Baseline
        fig1.add_hline(
            y=baseline_peak, 
            line_dash="dot", 
            line_color="red", 
            annotation_text=f"Baseline Piek: {baseline_peak:.2f} kW",
            annotation_position="top right"
        )
        
        # Pieklijn Geoptimaliseerd
        fig1.add_hline(
            y=optimal_peak, 
            line_dash="dash", 
            line_color="blue", 
            annotation_text=f"Geoptimaliseerde Piek: {optimal_peak:.2f} kW",
            annotation_position="bottom right"
        )
        
        fig1.update_layout(
            title_text="Nettovraag vóór en ná Peak Shaving Optimalisatie",
            xaxis_title="Tijd",
            yaxis_title="Vermogen (kW)",
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig1, use_container_width=True)

    # 2. Batterij SOC en Laad/Ontlaad Plot
    with tab2:
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                             subplot_titles=("Batterij SOC (State of Charge)", "Laad-/Ontlaadvermogen (Acties)"),
                             vertical_spacing=0.1)

        # SOC (Row 1)
        fig2.add_trace(go.Scatter(
            x=df_results.index, 
            y=df_results['SOC_opt'], 
            mode='lines', 
            name='SOC (kWh)', 
            line=dict(color='purple', width=2)
        ), row=1, col=1)
        fig2.update_yaxes(title_text="SOC (kWh)", range=[0, BATTERY_CAP_KWH * 1.05], row=1, col=1)

        # Laden (Row 2)
        fig2.add_trace(go.Scatter(
            x=df_results.index, 
            y=df_results['P_charge_opt'], 
            mode='lines', 
            name='Laden (kW)', 
            fill='tozeroy',
            fillcolor='rgba(0, 128, 0, 0.5)', 
            line=dict(color='darkgreen')
        ), row=2, col=1)
        
        # Ontladen (Row 2)
        fig2.add_trace(go.Scatter(
            x=df_results.index, 
            y=-df_results['P_discharge_opt'], # Negatief om ontladen aan te geven
            mode='lines', 
            name='Ontladen (kW)', 
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.5)', 
            line=dict(color='red')
        ), row=2, col=1)
        
        fig2.update_yaxes(title_text="Vermogen (kW)", range=[-MAX_P_KW * 1.1, MAX_P_KW * 1.1], row=2, col=1)
        fig2.update_layout(
            title_text="Batterij Gedrag",
            hovermode="x unified",
            height=600
        )
        st.plotly_chart(fig2, use_container_width=True)

    # 3. Nettovraag en Prijzen Plot
    with tab3:
        fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                             subplot_titles=("Baseline Nettovraag (Vraag-Aanbod)", "Day-Ahead Prijzen"),
                             vertical_spacing=0.1,
                             specs=[[{"secondary_y": False}], [{"secondary_y": True}]])
        
        # Baseline Nettovraag (Row 1)
        fig3.add_trace(go.Bar(
            x=df_data.index,
            y=df_data['Nettovraag_Baseline'],
            name='Nettovraag (kW)',
            marker_color='lightblue'
        ), row=1, col=1)
        fig3.update_yaxes(title_text="Nettovraag (kW)", row=1, col=1)

        # Prijzen (Row 2)
        fig3.add_trace(go.Scatter(
            x=df_data.index,
            y=df_data['Prijs_verbruik'],
            mode='lines',
            name='Verbruiksprijs (€/kWh)',
            line=dict(color='green')
        ), row=2, col=1)

        fig3.add_trace(go.Scatter(
            x=df_data.index,
            y=df_data['Prijs_injectie'],
            mode='lines',
            name='Injectieprijs (€/kWh)',
            line=dict(color='orange')
        ), row=2, col=1)
        
        fig3.update_yaxes(title_text="Prijs (€/kWh)", row=2, col=1)
        fig3.update_layout(
            title_text="Basis Input Data: Vraag en Prijzen",
            hovermode="x unified",
            height=600
        )
        st.plotly_chart(fig3, use_container_width=True)

else:
    st.error("Kon de optimalisatie niet uitvoeren. Controleer de invoerparameters.")
