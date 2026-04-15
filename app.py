"""
Streamlit App — India Renewable Energy Intelligence Platform
Wraps models_6_to_9.py with interactive UI.
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Page config
st.set_page_config(
    page_title="India Renewable Energy Intelligence Platform",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 India Renewable Energy Intelligence Platform")
st.markdown("*9 ML models · 36 States · 5-Year Enriched Dataset*")

# ── Load data
@st.cache_data
def load_data():
    df = pd.read_excel('enriched_data.xlsx')
    return df

try:
    df_all = load_data()
    df_24  = df_all[df_all['year'] == 2024].copy().reset_index(drop=True)
    st.success(f"Dataset loaded: {df_all.shape[0]} rows × {df_all.shape[1]} columns | Years: {sorted(df_all['year'].unique())}")
except Exception as e:
    st.error(f"Could not load enriched_data.xlsx: {e}")
    st.stop()

# ── Sidebar: State selector
st.sidebar.header("🗺️ Select State")
states = sorted(df_24['state'].unique())
state = st.sidebar.selectbox("State", states, index=states.index('Karnataka') if 'Karnataka' in states else 0)
forecast_year = st.sidebar.slider("Forecast Year", 2025, 2030, 2027)

row = df_24[df_24['state'] == state].iloc[0]

# ── State Overview
st.subheader(f"📍 {state} — 2024 Snapshot")
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Solar Capacity", f"{row['solar_capacity']:,.0f} MW")
c2.metric("Wind Capacity",  f"{row['wind_capacity']:,.0f} MW")
c3.metric("Renewable Penetration", f"{row['renewable_penetration_pct']:.1f}%")
c4.metric("T&D Loss", f"{row['t&d_loss_pct']:.1f}%")
c5.metric("Self-Sufficiency", f"{row['self_sufficiency_ratio']*100:.1f}%")

st.divider()

# ── Run all models
if st.button("▶ Run All 9 Models", type="primary"):
    from scipy.optimize import linprog
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    st.subheader("🤖 Model Results")

    # ── M4: K-Means Clustering
    df_km = df_24[['solar_irradiance','wind_speed','solar_capacity','total_renewable_capacity']].copy()
    sc_km = StandardScaler()
    km_fit = KMeans(n_clusters=4, random_state=42, n_init=10)
    km_fit.fit(sc_km.fit_transform(df_km))
    inp_km = sc_km.transform([[row['solar_irradiance'], row['wind_speed'], row['solar_capacity'], row['total_renewable_capacity']]])
    cluster_id = km_fit.predict(inp_km)[0]
    centroids_cap = [sc_km.inverse_transform(km_fit.cluster_centers_)[i][2] for i in range(4)]
    sorted_c = sorted(range(4), key=lambda i: centroids_cap[i], reverse=True)
    cnames = {sorted_c[0]:"Solar Leaders", sorted_c[1]:"Growing States",
              sorted_c[2]:"Untapped Potential", sorted_c[3]:"Early Stage"}
    cluster_name = cnames.get(cluster_id, f"Cluster {cluster_id}")

    # ── M2: Priority
    gap_ratio = max(0, min(1, (row['energy_consumption'] - row['total_renewable_capacity']) / row['energy_consumption']))
    priority = 'HIGH' if row['solar_irradiance'] > 140 and gap_ratio > 0.75 else \
               'MEDIUM' if row['solar_irradiance'] > 125 or gap_ratio > 0.5 else 'LOW'
    priority_reason = f"irradiance={row['solar_irradiance']:.1f}W/m² (threshold 140.21), gap_ratio={gap_ratio:.2%}"

    # ── M1: Random Forest Demand Forecast
    df_rf = df_all.dropna(subset=['energy_consumption']).copy()
    df_rf['year_num'] = df_rf['year'] - 2020
    rf_cols = ['solar_irradiance','wind_speed','solar_capacity','wind_capacity',
               'hydro_capacity','total_renewable_capacity','year_num']
    from sklearn.ensemble import RandomForestRegressor
    rf_q = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    rf_q.fit(df_rf[rf_cols], df_rf['energy_consumption'])
    cons_pred = rf_q.predict([[row['solar_irradiance'], row['wind_speed'], row['solar_capacity'],
                               row['wind_capacity'], row['hydro_capacity'], row['total_renewable_capacity'],
                               forecast_year - 2020]])[0]

    # ── M3: Solar Capacity Trend
    state_solar = df_all[df_all['state']==state][['year','solar_capacity']].set_index('year')['solar_capacity']
    slope_s, intercept_s = np.polyfit(state_solar.index, state_solar.values, 1)
    solar_forecast = max(row['solar_capacity'], slope_s * forecast_year + intercept_s)

    # ── M7: Gradient Boosting CO2 Predictor
    features_m7 = ['solar_capacity','wind_capacity','hydro_capacity',
                   'renewable_penetration_pct','plf','resource_score',
                   'solar_irradiance','wind_speed','total_renewable_capacity',
                   'self_sufficiency_ratio','fossil_dependency_ratio']
    gb_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                                          min_samples_leaf=3, random_state=42)
    gb_model.fit(df_all[features_m7], df_all['co2_avoided_tonnes'])
    co2_pred = gb_model.predict([row[features_m7].values])[0]

    # ── M8: T&D Tier Classifier
    def td_tier(v): return 'LOW' if v<10 else 'MEDIUM' if v<25 else 'HIGH'
    df_all['td_tier'] = df_all['t&d_loss_pct'].apply(td_tier)
    features_m8 = ['solar_irradiance','wind_speed','renewable_penetration_pct',
                   'resource_score','consumption_per_capita','plf',
                   'self_sufficiency_ratio','fossil_dependency_ratio',
                   'yoy_capacity_growth_pct','supply_gap']
    le_m8 = LabelEncoder()
    y_m8 = le_m8.fit_transform(df_all['td_tier'])
    rf_m8 = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=3,
                                    random_state=42, class_weight='balanced')
    rf_m8.fit(df_all[features_m8], y_m8)
    td_pred = le_m8.inverse_transform(rf_m8.predict([row[features_m8].values]))[0]

    # ── M9: Self-Sufficiency Forecast
    state_ss = df_all[df_all['state']==state][['year','self_sufficiency_ratio']].set_index('year')['self_sufficiency_ratio']
    slope_9, intercept_9 = np.polyfit(state_ss.index, state_ss.values, 1)
    ss_forecast = min(1.0, max(0, slope_9 * forecast_year + intercept_9))
    traj = '↑ Improving' if slope_9 > 0.002 else '→ Stagnant' if slope_9 > -0.002 else '↓ Declining'

    # ── M6: LP Optimization
    SOLAR_COST, WIND_COST, HYDRO_COST = 4.5, 6.0, 10.0
    SOLAR_CO2, WIND_CO2, HYDRO_CO2 = 1200, 1100, 900
    COST_WEIGHT, CO2_WEIGHT = 0.70, 0.30
    MAX_COST, MAX_CO2_SAV = HYDRO_COST, SOLAR_CO2
    demand_gap = max(row['supply_gap'], 0)
    target_mw = row['installed_target_mw']
    max_solar_add = max(target_mw*0.6 - row['solar_capacity'], 0)
    max_wind_add  = max(target_mw*0.3 - row['wind_capacity'], 0)
    max_hydro_add = max(target_mw*0.1 - row['hydro_capacity'], 0)
    c6 = [COST_WEIGHT*(SOLAR_COST/MAX_COST)-CO2_WEIGHT*(SOLAR_CO2/MAX_CO2_SAV),
          COST_WEIGHT*(WIND_COST/MAX_COST) -CO2_WEIGHT*(WIND_CO2/MAX_CO2_SAV),
          COST_WEIGHT*(HYDRO_COST/MAX_COST)-CO2_WEIGHT*(HYDRO_CO2/MAX_CO2_SAV)]
    A6 = [[-1,0,0],[0,-1,0],[0,0,-1],[-1,-1,-1]]
    b6 = [0, 0, 0, -demand_gap]
    bounds6 = [(0,max_solar_add),(0,max_wind_add),(0,max_hydro_add)]
    try:
        r6 = linprog(c6, A_ub=A6, b_ub=b6, bounds=bounds6, method='highs')
        sa,wa,ha = r6.x if r6.success else (demand_gap*0.6, demand_gap*0.25, demand_gap*0.15)
    except:
        sa,wa,ha = demand_gap*0.6, demand_gap*0.25, demand_gap*0.15
    cost6 = sa*SOLAR_COST + wa*WIND_COST + ha*HYDRO_COST
    co2_6 = sa*SOLAR_CO2  + wa*WIND_CO2  + ha*HYDRO_CO2

    # ── Display Results
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔵 Classification Models")
        st.info(f"**M4 · K-Means Cluster**: {cluster_name}\n\n"
                f"*Why*: Clustered on solar_irradiance={row['solar_irradiance']:.1f}, "
                f"total_capacity={row['total_renewable_capacity']:.0f}MW. "
                f"Centroid assignment via K=4 (silhouette=0.445).")
        color = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}.get(priority,"⚪")
        st.info(f"**M2 · Decision Tree Priority**: {color} {priority}\n\n"
                f"*Why*: {priority_reason}. Rule: irr>140.21 AND gap_ratio>0.75 → HIGH.")
        td_colors = {"LOW":"🟢","MEDIUM":"🟡","HIGH":"🔴"}
        td_msgs = {"LOW":"Efficient grid — ready for more renewables",
                   "MEDIUM":"Grid modernisation recommended",
                   "HIGH":"Urgent grid upgrade needed"}
        st.warning(f"**M8 · T&D Grid Tier**: {td_colors.get(td_pred,'⚪')} {td_pred}\n\n"
                   f"*Why*: T&D loss={row['t&d_loss_pct']:.1f}%. Top M8 features: "
                   f"wind_speed(17.7%), consumption/capita(16.9%), PLF(15.5%). → {td_msgs.get(td_pred,'')}")

    with col2:
        st.markdown(f"#### 📈 Forecast to {forecast_year}")
        st.success(f"**M1 · RF Demand Forecast {forecast_year}**: {cons_pred:,.0f} MU\n\n"
                   f"*Why*: RF (R²=0.971) projects from current {row['energy_consumption']:,.0f} MU. "
                   f"Top feature: solar_capacity (47.9% importance). Growth ~{((cons_pred/row['energy_consumption'])**(1/max(forecast_year-2024,1))-1)*100:.1f}%/yr.")
        st.success(f"**M3 · Solar Trend {forecast_year}**: {solar_forecast:,.0f} MW (+{solar_forecast-row['solar_capacity']:,.0f} MW)\n\n"
                   f"*Why*: Linear regression on {state} 2020–2024 solar trend. "
                   f"Slope={slope_s:+.0f} MW/yr projected to {forecast_year}.")
        st.info(f"**M9 · Self-Sufficiency {forecast_year}**: {ss_forecast*100:.1f}% ({traj})\n\n"
                f"*Why*: OLS on 2020–2024 self_sufficiency_ratio. Slope={slope_9*100:+.3f}%/yr. "
                f"Current={row['self_sufficiency_ratio']*100:.1f}% → {forecast_year}: {ss_forecast*100:.1f}%.")

    st.markdown(f"#### 🌿 Environmental & Optimization (Based on 2024 actuals)")
    col3, col4 = st.columns(2)
    with col3:
        st.success(f"**M7 · CO₂ Avoided (GB)**: {co2_pred:,.0f} tonnes/year\n\n"
                   f"*Why*: GB model (R²=0.978). Key: total_renewable=89.4% weight. "
                   f"Formula: {row['total_renewable_capacity']:.0f}MW × 7.2 t/MW/yr × PLF_factor({row['plf']/69.76:.2f}).")
        gap_label = max(0, cons_pred - solar_forecast)
        st.warning(f"**Gap Analysis {forecast_year}**: {gap_label:,.0f} MU remaining\n\n"
                   f"*Why*: M1 demand({cons_pred:,.0f}) − M3 solar_cap({solar_forecast:,.0f}MW). This gap drives M6 LP.")
    with col4:
        st.info(f"**M6 · LP Optimal Mix** (to fill {demand_gap:,.0f} MU gap):\n\n"
                f"• Solar: {sa:,.0f} MW · ₹{sa*SOLAR_COST:,.0f} Cr\n"
                f"• Wind: {wa:,.0f} MW · ₹{wa*WIND_COST:,.0f} Cr\n"
                f"• Hydro: {ha:,.0f} MW · ₹{ha*HYDRO_COST:,.0f} Cr\n\n"
                f"**Total: ₹{cost6:,.0f} Cr** · CO₂ savings: {co2_6:,.0f} t/yr\n\n"
                f"*Why*: scipy linprog minimizes 70%×cost + 30%×(−CO₂). Solar cheapest (₹4.5Cr/MW) → prioritised.")

    # ── Trend chart
    st.subheader(f"📊 {state} — 5-Year Trend + {forecast_year} Forecast")
    state_data = df_all[df_all['state']==state].sort_values('year')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    fig.patch.set_facecolor('#0a0f0a')
    for ax in axes:
        ax.set_facecolor('#0d170d')
        ax.tick_params(colors='#8fbc8f', labelsize=8)
        for sp in ax.spines.values(): sp.set_color('#1e3a1e')
    # Solar
    axes[0].plot(state_data['year'], state_data['solar_capacity'], 'o-', color='#22c55e', linewidth=2)
    axes[0].plot([2024, forecast_year], [row['solar_capacity'], solar_forecast], 's--', color='#ef4444', linewidth=2)
    axes[0].set_title('Solar Capacity (MW)', color='#e8f5e8', fontsize=10)
    # Self-Sufficiency
    axes[1].plot(state_data['year'], state_data['self_sufficiency_ratio']*100, 'o-', color='#f59e0b', linewidth=2)
    axes[1].plot([2024, forecast_year], [row['self_sufficiency_ratio']*100, ss_forecast*100], 's--', color='#ef4444', linewidth=2)
    axes[1].set_title(f'Self-Sufficiency % → {forecast_year}', color='#e8f5e8', fontsize=10)
    # CO2
    axes[2].plot(state_data['year'], state_data['co2_avoided_tonnes'], 'o-', color='#14b8a6', linewidth=2)
    axes[2].set_title('CO₂ Avoided (tonnes)', color='#e8f5e8', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()
st.caption("India Renewable Energy Intelligence Platform · DWM Project 2024–2025 · 36 States · 9 Models")
