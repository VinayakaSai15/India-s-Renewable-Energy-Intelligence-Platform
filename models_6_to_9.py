"""
ADVANCED MODELS — Enriched Dataset (180 rows, 5 years, 36 states, 41 features)
================================================================================

MODEL 6  (YOUR IDEA):  Optimization-Based Energy Planning
         "How do we meet demand at lowest cost and least emissions?"
         → Uses scipy linprog to find optimal solar/wind/hydro mix per state

MODEL 7  (NEW INSIGHT): CO2 Emissions Avoidance Predictor
         "Which states are saving the most carbon, and who's falling behind?"
         → Gradient Boosting Regression on co2_avoided_tonnes

MODEL 8  (NEW INSIGHT): Grid Efficiency & T&D Loss Classifier
         "Can we predict which states have dangerously high transmission losses?"
         → Random Forest Classifier on t&d_loss_pct tiers

MODEL 9  (NEW INSIGHT): Renewable Self-Sufficiency Forecaster
         "Which states will become energy self-sufficient by 2030?"
         → Multi-year projection using self_sufficiency_ratio trend
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import linprog
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, classification_report, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
# # df_all = pd.read_excel('enriched_data.xlsx')
# def load_data():
#     import pandas as pd
#     return pd.read_excel('enriched_data.xlsx')
# df_24  = df_all[df_all['year'] == 2024].copy().reset_index(drop=True)
# df_23  = df_all[df_all['year'] == 2023].copy().reset_index(drop=True)
import pandas as pd

def load_data():
    return pd.read_excel('enriched_data.xlsx')

def get_2024_data():
    df_all = load_data()
    df_24 = df_all[df_all['year'] == 2024].copy().reset_index(drop=True)
    return df_24
print("=" * 70)
print("ADVANCED MODELS — Enriched Dataset")
print("=" * 70)
print(f"Dataset: {df_all.shape[0]} rows × {df_all.shape[1]} columns")
print(f"Years  : {sorted(df_all['year'].unique())}")
print(f"States : {df_all['state'].nunique()}")

# ═══════════════════════════════════════════════════════════════════════
# MODEL 6: OPTIMIZATION-BASED ENERGY PLANNING
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MODEL 6: OPTIMIZATION-BASED ENERGY PLANNING")
print("=" * 70)

print("""
CONCEPT — Linear Programming Optimization
------------------------------------------
The goal: For each state, find the OPTIMAL mix of Solar + Wind + Hydro
that meets the energy demand while:
  1. Minimizing COST  (solar cheapest, wind moderate, hydro expensive to build)
  2. Minimizing CO2   (all renewables help, but different amounts)
  3. Meeting DEMAND   (solar + wind + hydro >= supply_gap)
  4. Respecting LIMITS (can't exceed resource potential per state)

This is called Linear Programming (LP). We define:
  x = [solar_to_add_MW, wind_to_add_MW, hydro_to_add_MW]
  Minimize: cost_vector · x
  Subject to: constraint_matrix · x <= bounds

scipy.optimize.linprog() solves this in milliseconds.
""")

# Cost assumptions (₹ crore per MW installed, approximate 2024 India values)
SOLAR_COST  = 4.5   # ₹ Crore/MW  (cheapest, rapidly falling)
WIND_COST   = 6.0   # ₹ Crore/MW  (moderate)
HYDRO_COST  = 10.0  # ₹ Crore/MW  (most expensive, civil works)

# CO2 savings (tonnes CO2 avoided per MW per year vs coal baseline)
SOLAR_CO2   = 1200  # tonnes/MW/year
WIND_CO2    = 1100  # tonnes/MW/year
HYDRO_CO2   = 900   # tonnes/MW/year

# Weight for multi-objective: 70% cost, 30% emissions
COST_WEIGHT = 0.70
CO2_WEIGHT  = 0.30

# Normalize for combined objective
MAX_COST    = HYDRO_COST
MAX_CO2_SAV = SOLAR_CO2

optimization_results = []

for _, row in df_24.iterrows():
    state         = row['state']
    demand_gap    = max(row['supply_gap'], 0)
    resource_score= row['resource_score']  # 0-100 how good the resources are

    # Current installed
    curr_solar = row['solar_capacity']
    curr_wind  = row['wind_capacity']
    curr_hydro = row['hydro_capacity']
    target_mw  = row['installed_target_mw']

    # How much MORE capacity can be added (gap to target)
    max_solar_add = max(target_mw * 0.6 - curr_solar, 0)  # solar gets 60% of target
    max_wind_add  = max(target_mw * 0.3 - curr_wind,  0)  # wind gets 30%
    max_hydro_add = max(target_mw * 0.1 - curr_hydro, 0)  # hydro gets 10%

    if demand_gap < 10:
        # Already close to sufficient
        optimization_results.append({
            'state': state,
            'demand_gap_mw': round(demand_gap, 1),
            'opt_solar_add': 0, 'opt_wind_add': 0, 'opt_hydro_add': 0,
            'total_add_mw': 0,
            'estimated_cost_crore': 0,
            'co2_saving_tonnes': 0,
            'status': 'Already Sufficient'
        })
        continue

    # ── Objective: minimize weighted cost + CO2 penalty
    # linprog minimizes c·x, so CO2 savings = negative (we want to maximize savings)
    c = [
        COST_WEIGHT * (SOLAR_COST / MAX_COST) - CO2_WEIGHT * (SOLAR_CO2 / MAX_CO2_SAV),
        COST_WEIGHT * (WIND_COST  / MAX_COST) - CO2_WEIGHT * (WIND_CO2  / MAX_CO2_SAV),
        COST_WEIGHT * (HYDRO_COST / MAX_COST) - CO2_WEIGHT * (HYDRO_CO2 / MAX_CO2_SAV),
    ]

    # ── Inequality constraints: A_ub @ x <= b_ub
    A_ub = [
        [-1,  0,  0],   # -solar <= -0 (solar >= 0, handled by bounds)
        [ 0, -1,  0],   # -wind  >= 0
        [ 0,  0, -1],   # -hydro >= 0
    ]
    b_ub = [0, 0, 0]

    # ── Equality constraint: total generation >= demand_gap
    # Reformulated as: -solar - wind - hydro <= -demand_gap
    A_ub.append([-1, -1, -1])
    b_ub.append(-demand_gap)

    # ── Variable bounds: 0 to max additional
    bounds = [(0, max_solar_add), (0, max_wind_add), (0, max_hydro_add)]

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            solar_add, wind_add, hydro_add = result.x
            cost = solar_add * SOLAR_COST + wind_add * WIND_COST + hydro_add * HYDRO_COST
            co2  = solar_add * SOLAR_CO2  + wind_add * WIND_CO2  + hydro_add * HYDRO_CO2
            status = 'Optimal'
        else:
            # Fallback: simple proportional split
            solar_add = demand_gap * 0.6
            wind_add  = demand_gap * 0.25
            hydro_add = demand_gap * 0.15
            cost = solar_add * SOLAR_COST + wind_add * WIND_COST + hydro_add * HYDRO_COST
            co2  = solar_add * SOLAR_CO2  + wind_add * WIND_CO2  + hydro_add * HYDRO_CO2
            status = 'Fallback'
    except:
        solar_add = demand_gap * 0.6
        wind_add  = demand_gap * 0.25
        hydro_add = demand_gap * 0.15
        cost = solar_add * SOLAR_COST + wind_add * WIND_COST + hydro_add * HYDRO_COST
        co2  = solar_add * SOLAR_CO2  + wind_add * WIND_CO2  + hydro_add * HYDRO_CO2
        status = 'Fallback'

    optimization_results.append({
        'state'               : state,
        'demand_gap_mw'       : round(demand_gap, 1),
        'opt_solar_add'       : round(solar_add, 1),
        'opt_wind_add'        : round(wind_add, 1),
        'opt_hydro_add'       : round(hydro_add, 1),
        'total_add_mw'        : round(solar_add + wind_add + hydro_add, 1),
        'estimated_cost_crore': round(cost, 1),
        'co2_saving_tonnes'   : round(co2, 0),
        'status'              : status
    })

opt_df = pd.DataFrame(optimization_results)
opt_df.to_csv('model6_optimization_results.csv', index=False)

print("Optimization Results — Top 10 states by investment needed:")
print(opt_df.nlargest(10,'estimated_cost_crore')[
    ['state','demand_gap_mw','opt_solar_add','opt_wind_add','opt_hydro_add',
     'estimated_cost_crore','co2_saving_tonnes','status']
].to_string(index=False))

# ── Chart 6a: Optimal energy mix per state (stacked bar)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

top15 = opt_df[opt_df['status']!='Already Sufficient'].nlargest(15,'total_add_mw')
x = np.arange(len(top15))
ax = axes[0]
ax.bar(x, top15['opt_solar_add'], label='Solar to Add (MW)', color='#EF9F27', alpha=0.9)
ax.bar(x, top15['opt_wind_add'],  bottom=top15['opt_solar_add'],
       label='Wind to Add (MW)', color='#534AB7', alpha=0.9)
ax.bar(x, top15['opt_hydro_add'],
       bottom=top15['opt_solar_add'] + top15['opt_wind_add'],
       label='Hydro to Add (MW)', color='#185FA5', alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels([s[:13] for s in top15['state']], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Capacity to Add (MW)', fontsize=11)
ax.set_title('Optimal Energy Mix to Fill Demand Gap\n(Top 15 States)', fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, axis='y', alpha=0.3)

# Chart 6b: Cost vs CO2 savings scatter
ax2 = axes[1]
valid = opt_df[opt_df['status']!='Already Sufficient']
scatter = ax2.scatter(valid['estimated_cost_crore'], valid['co2_saving_tonnes']/1000,
                      c=valid['demand_gap_mw'], cmap='RdYlGn_r',
                      s=100, alpha=0.8, edgecolors='white')
plt.colorbar(scatter, ax=ax2, label='Demand Gap (MW)')
for _, row in valid[valid['estimated_cost_crore'] > 50000].iterrows():
    ax2.annotate(row['state'][:12], (row['estimated_cost_crore'], row['co2_saving_tonnes']/1000),
                 fontsize=7, xytext=(3,3), textcoords='offset points')
ax2.set_xlabel('Estimated Investment (₹ Crore)', fontsize=11)
ax2.set_ylabel('CO2 Saved (000 Tonnes/year)', fontsize=11)
ax2.set_title('Investment vs CO2 Savings\n(Bigger bubble = larger demand gap)',
              fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.suptitle('Model 6: Optimization-Based Energy Planning', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('model6_optimization.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nChart saved → model6_optimization.png")
print("Results saved → model6_optimization_results.csv")

# ═══════════════════════════════════════════════════════════════════════
# MODEL 7: CO2 AVOIDANCE PREDICTOR (Gradient Boosting)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MODEL 7: CO2 EMISSIONS AVOIDANCE PREDICTOR — Gradient Boosting")
print("=" * 70)

print("""
CONCEPT — Gradient Boosting Regression
----------------------------------------
Gradient Boosting builds trees SEQUENTIALLY. Each new tree corrects the
errors of the previous one — like a student learning from mistakes.
It's more powerful than Random Forest for regression when data has
complex patterns. Ideal here because CO2 avoidance depends on multiple
interacting factors: capacity, PLF, resource score, and penetration %.

Why this insight matters:
  CO2 avoided = tonnes of carbon NOT emitted because renewables replaced coal.
  Predicting this lets states measure their environmental contribution and
  set science-based carbon reduction targets.
""")

features_m7 = ['solar_capacity', 'wind_capacity', 'hydro_capacity',
                'renewable_penetration_pct', 'plf', 'resource_score',
                'solar_irradiance', 'wind_speed', 'total_renewable_capacity',
                'self_sufficiency_ratio', 'fossil_dependency_ratio']
target_m7 = 'co2_avoided_tonnes'

X_m7 = df_all[features_m7]
y_m7 = df_all[target_m7]

gb_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,   # small learning rate = more robust
    max_depth=4,
    min_samples_leaf=3,
    random_state=42
)
gb_model.fit(X_m7, y_m7)

cv_r2_m7 = cross_val_score(gb_model, X_m7, y_m7, cv=5, scoring='r2')
y_pred_m7 = gb_model.predict(X_m7)
train_r2_m7 = r2_score(y_m7, y_pred_m7)
mae_m7 = mean_absolute_error(y_m7, y_pred_m7)

print(f"Gradient Boosting Results:")
print(f"  Training R²      : {train_r2_m7:.3f}")
print(f"  Cross-val R²     : {cv_r2_m7.mean():.3f} ± {cv_r2_m7.std():.3f}")
print(f"  Training MAE     : {mae_m7:,.0f} tonnes")

imp_m7 = pd.Series(gb_model.feature_importances_, index=features_m7).sort_values(ascending=False)
print(f"\nFeature Importance:")
for feat, imp in imp_m7.items():
    bar = "█" * int(imp * 50)
    print(f"  {feat:35s} {bar} {imp*100:.1f}%")

# Add predictions to 2024 data
df_24['co2_predicted'] = gb_model.predict(df_24[features_m7])

# Find underperformers: high capacity but low CO2 savings (efficiency issue)
df_24['co2_efficiency'] = df_24['co2_avoided_tonnes'] / (df_24['total_renewable_capacity'] + 1)
print("\nTop CO2 savers (2024):")
print(df_24.nlargest(5,'co2_avoided_tonnes')[['state','co2_avoided_tonnes','renewable_penetration_pct']].to_string(index=False))
print("\nCO2 underperformers (high capacity, lower-than-predicted savings):")
df_24['co2_gap'] = df_24['co2_predicted'] - df_24['co2_avoided_tonnes']
print(df_24.nlargest(5,'co2_gap')[['state','co2_avoided_tonnes','co2_predicted','co2_gap']].to_string(index=False))

# Chart 7
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.scatter(y_m7, y_pred_m7, color='#1D9E75', alpha=0.6, s=60, edgecolors='white')
max_v = max(y_m7.max(), y_pred_m7.max())
ax.plot([0, max_v], [0, max_v], 'r--', linewidth=1.5, label='Perfect')
ax.set_xlabel('Actual CO2 Avoided (tonnes)', fontsize=11)
ax.set_ylabel('Predicted CO2 Avoided (tonnes)', fontsize=11)
ax.set_title(f'Model 7: CO2 Predictor\nR² = {train_r2_m7:.3f}', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

ax2 = axes[1]
imp_top = imp_m7.head(6)
colors = ['#1D9E75' if v > 0.1 else '#9FE1CB' for v in imp_top.values]
bars = ax2.barh(imp_top.index, imp_top.values, color=colors, alpha=0.9)
ax2.set_xlabel('Feature Importance', fontsize=11)
ax2.set_title('What Drives CO2 Avoidance?', fontweight='bold', fontsize=12)
for bar, val in zip(bars, imp_top.values):
    ax2.text(val+0.003, bar.get_y()+bar.get_height()/2,
             f'{val*100:.1f}%', va='center', fontsize=9)
ax2.grid(True, axis='x', alpha=0.3)

plt.suptitle('Model 7: CO2 Emissions Avoidance Predictor', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('model7_co2_predictor.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nChart saved → model7_co2_predictor.png")

# ═══════════════════════════════════════════════════════════════════════
# MODEL 8: GRID EFFICIENCY CLASSIFIER (T&D Loss Tier Prediction)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MODEL 8: GRID EFFICIENCY CLASSIFIER — T&D Loss Prediction")
print("=" * 70)

print("""
CONCEPT — T&D Loss Classification
-----------------------------------
Transmission & Distribution (T&D) loss = electricity generated but LOST
in the grid before reaching consumers. India's national average is ~17%.
High T&D loss (>25%) = inefficient grid = wasted renewable energy.

Why it matters for your project:
  Even if a state builds solar capacity, high T&D losses mean that energy
  never reaches consumers. Predicting which states have high/medium/low
  losses helps identify where grid upgrades are needed BEFORE building capacity.

We classify each state-year as:
  LOW    = T&D < 10%   (efficient, grid-ready for more renewables)
  MEDIUM = 10–25%      (needs modernisation)
  HIGH   = > 25%       (urgent grid upgrade needed)
""")

# Create T&D loss tiers
def td_tier(val):
    if val < 10:  return 'LOW'
    elif val < 25: return 'MEDIUM'
    else:          return 'HIGH'

df_all['td_tier'] = df_all['t&d_loss_pct'].apply(td_tier)
print("T&D Loss tier distribution:")
print(df_all['td_tier'].value_counts())

features_m8 = ['solar_irradiance', 'wind_speed', 'renewable_penetration_pct',
                'resource_score', 'consumption_per_capita', 'plf',
                'self_sufficiency_ratio', 'fossil_dependency_ratio',
                'yoy_capacity_growth_pct', 'supply_gap']

X_m8 = df_all[features_m8]
le_m8 = LabelEncoder()
y_m8 = le_m8.fit_transform(df_all['td_tier'])

rf_m8 = RandomForestClassifier(
    n_estimators=200, max_depth=5,
    min_samples_leaf=3, random_state=42,
    class_weight='balanced'
)
rf_m8.fit(X_m8, y_m8)

cv_m8 = cross_val_score(rf_m8, X_m8, y_m8, cv=5, scoring='accuracy')
train_acc_m8 = (rf_m8.predict(X_m8) == y_m8).mean()

print(f"\nRandom Forest Classifier Results:")
print(f"  Training Accuracy : {train_acc_m8*100:.1f}%")
print(f"  Cross-val Accuracy: {cv_m8.mean()*100:.1f}% ± {cv_m8.std()*100:.1f}%")

print("\nClassification Report:")
y_pred_m8 = rf_m8.predict(X_m8)
print(classification_report(y_m8, y_pred_m8, target_names=le_m8.classes_))

imp_m8 = pd.Series(rf_m8.feature_importances_, index=features_m8).sort_values(ascending=False)
print("Top features predicting T&D loss tier:")
for feat, imp in imp_m8.head(5).items():
    print(f"  {feat:35s} {imp*100:.1f}%")

# 2024 predictions
df_24['td_tier_predicted'] = le_m8.inverse_transform(rf_m8.predict(df_24[features_m8]))
print("\n2024 States predicted as HIGH T&D loss (urgent grid upgrade):")
high_td = df_24[df_24['td_tier_predicted']=='HIGH'][['state','t&d_loss_pct','renewable_penetration_pct']]
print(high_td.to_string(index=False))

# Chart 8
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# T&D loss distribution by tier
ax = axes[0]
colors_td = {'LOW': '#1D9E75', 'MEDIUM': '#EF9F27', 'HIGH': '#E24B4A'}
for tier, color in colors_td.items():
    subset = df_all[df_all['td_tier'] == tier]['t&d_loss_pct']
    ax.hist(subset, bins=10, color=color, alpha=0.7, label=f'{tier} ({len(subset)})', edgecolor='white')
ax.axvline(x=10, color='orange', linestyle='--', linewidth=1.5, label='10% threshold')
ax.axvline(x=25, color='red',    linestyle='--', linewidth=1.5, label='25% threshold')
ax.set_xlabel('T&D Loss (%)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('T&D Loss Distribution by Tier', fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Feature importance for T&D prediction
ax2 = axes[1]
imp_top8 = imp_m8.head(6)
bar_colors8 = ['#E24B4A' if v > 0.15 else '#F5C4B3' for v in imp_top8.values]
bars8 = ax2.barh(imp_top8.index, imp_top8.values, color=bar_colors8, alpha=0.9)
ax2.set_xlabel('Feature Importance', fontsize=11)
ax2.set_title('What Predicts Grid Efficiency?', fontweight='bold', fontsize=12)
for bar, val in zip(bars8, imp_top8.values):
    ax2.text(val+0.003, bar.get_y()+bar.get_height()/2,
             f'{val*100:.1f}%', va='center', fontsize=9)
ax2.grid(True, axis='x', alpha=0.3)

plt.suptitle('Model 8: Grid Efficiency & T&D Loss Classifier', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('model8_grid_efficiency.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nChart saved → model8_grid_efficiency.png")

# ═══════════════════════════════════════════════════════════════════════
# MODEL 9: RENEWABLE SELF-SUFFICIENCY FORECASTER (2025–2030)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MODEL 9: RENEWABLE SELF-SUFFICIENCY FORECASTER (2025–2030)")
print("=" * 70)

print("""
CONCEPT — Self-Sufficiency Trend Projection
---------------------------------------------
self_sufficiency_ratio = renewable_capacity / energy_consumption
  → 1.0 means 100% energy self-sufficient from renewables
  → 0.0 means entirely fossil dependent

We fit a linear trend per state on 5 years (2020–2024) and project to 2030.
This answers the key policy question:
  "Which states will become renewable self-sufficient by 2030,
   and which ones won't even come close?"

States crossing ratio=1.0 are 'Energy Independent' — a landmark achievement.
""")

suff_pivot = df_all.pivot_table(
    index='state', columns='year', values='self_sufficiency_ratio')

years_hist = np.array([2020, 2021, 2022, 2023, 2024])
future_years = np.array([2025, 2026, 2027, 2028, 2029, 2030])

suff_results = []
state_trends = {}

for state in suff_pivot.index:
    vals = suff_pivot.loc[state, years_hist].values.astype(float)
    slope, intercept = np.polyfit(years_hist, vals, 1)
    state_trends[state] = (slope, intercept)

    future_vals = {yr: max(0, min(1, slope * yr + intercept)) for yr in future_years}
    current_val = vals[-1]

    # When will it reach 0.5 (50% self-sufficient)?
    if slope > 0:
        yr_50 = (0.5 - intercept) / slope
        yr_100 = (1.0 - intercept) / slope
    else:
        yr_50 = None
        yr_100 = None

    entry = {
        'state': state,
        'ratio_2024': round(current_val, 4),
        'annual_improvement': round(slope, 5),
        'ratio_2025': round(future_vals[2025], 4),
        'ratio_2027': round(future_vals[2027], 4),
        'ratio_2030': round(future_vals[2030], 4),
        'yr_reach_50pct': round(yr_50) if yr_50 and 2024 < yr_50 < 2060 else 'Already reached' if current_val >= 0.5 else 'Not in horizon',
        'yr_reach_100pct': round(yr_100) if yr_100 and 2024 < yr_100 < 2100 else 'Already reached' if current_val >= 1.0 else 'Not in horizon',
        'trajectory': 'Improving' if slope > 0.002 else 'Stagnant' if slope > -0.002 else 'Declining'
    }
    suff_results.append(entry)

suff_df = pd.DataFrame(suff_results).sort_values('ratio_2030', ascending=False)
suff_df.to_csv('model9_self_sufficiency_forecast.csv', index=False)

print("\nSelf-Sufficiency Forecast by 2030 (Top 10):")
print(suff_df[['state','ratio_2024','ratio_2030','annual_improvement',
               'yr_reach_50pct','trajectory']].head(10).to_string(index=False))

print("\nTrajectory distribution:")
print(suff_df['trajectory'].value_counts())

# Chart 9: Trajectory lines for top 8 most interesting states
interesting = pd.concat([
    suff_df.nlargest(4, 'ratio_2030'),
    suff_df[suff_df['trajectory']=='Improving'].nlargest(4, 'annual_improvement')
]).drop_duplicates(subset='state').head(8)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
all_years_plot = np.concatenate([years_hist, future_years])

for idx, (_, row) in enumerate(interesting.iterrows()):
    state = row['state']
    ax = axes[idx]
    slope, intercept = state_trends[state]

    hist_vals  = suff_pivot.loc[state, years_hist].values
    future_vals = np.clip(slope * future_years + intercept, 0, 1)

    ax.plot(years_hist, hist_vals, 'o-', color='#185FA5', linewidth=2,
            markersize=6, label='Actual')
    ax.plot(future_years, future_vals, 's--', color='#E24B4A', linewidth=2,
            markersize=5, label='Forecast', alpha=0.85)
    ax.axhline(y=0.5, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='50%')
    ax.axhline(y=1.0, color='green',  linestyle=':', linewidth=1, alpha=0.7, label='100%')
    ax.axvline(x=2024.5, color='#ccc', linestyle='-', linewidth=0.8)
    ax.set_title(state[:18], fontweight='bold', fontsize=9)
    ax.set_ylim(0, min(1.1, max(hist_vals.max(), future_vals.max()) * 1.2))
    ax.set_ylabel('Self-Sufficiency Ratio', fontsize=8)
    ax.set_xticks([2020, 2022, 2024, 2026, 2028, 2030])
    ax.set_xticklabels(['20','22','24','26','28','30'], fontsize=7)
    ax.grid(True, alpha=0.2)
    if idx == 0:
        ax.legend(fontsize=7)

plt.suptitle('Model 9: Renewable Self-Sufficiency Forecast (2025–2030)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('model9_self_sufficiency.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nChart saved → model9_self_sufficiency.png")
print("Results saved → model9_self_sufficiency_forecast.csv")

# ═══════════════════════════════════════════════════════════════════════
# COMBINED PREDICTION ENGINE — ALL 9 MODELS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("UNIFIED PREDICTION ENGINE — ALL 9 MODELS")
print("=" * 70)

def full_predict(
    state_name,
    solar_irradiance, wind_speed,
    solar_capacity, wind_capacity=0, hydro_capacity=0,
    energy_consumption=None, population=None,
    plf=63.0, td_loss_pct=15.0, resource_score=65.0,
    installed_target_mw=None, year=2025
):
    """
    Run ALL 9 models on any state's input data.
    Returns cluster, priority, consumption forecast, capacity forecast,
    gap, optimization plan, CO2 prediction, T&D tier, and self-sufficiency forecast.
    """
    # Derived inputs
    renewable_total = solar_capacity + wind_capacity + hydro_capacity
    consumption = energy_consumption if energy_consumption else 20000
    supply_gap  = max(consumption - renewable_total, 0)
    self_suff   = min(renewable_total / consumption, 1.0) if consumption > 0 else 0
    fossil_dep  = 1 - self_suff
    renew_pct   = (renewable_total / consumption * 100) if consumption > 0 else 0
    cons_per_cap= consumption / population if population else 0
    target_mw   = installed_target_mw if installed_target_mw else renewable_total * 2

    print("\n" + "═"*70)
    print(f"  FULL PREDICTION REPORT: {state_name.upper()}")
    print("═"*70)
    print(f"  solar_irradiance  = {solar_irradiance} W/m²")
    print(f"  wind_speed        = {wind_speed} m/s")
    print(f"  solar_capacity    = {solar_capacity} MW")
    print(f"  wind_capacity     = {wind_capacity} MW")
    print(f"  hydro_capacity    = {hydro_capacity} MW")
    print(f"  energy_consumption= {consumption:,} MU")
    print(f"  PLF               = {plf}%")
    print(f"  T&D loss          = {td_loss_pct}%")

    # ── Models 4,2,1,3 (from previous file — re-run logic inline)
    from sklearn.cluster import KMeans
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestRegressor

    # Quick retrain on enriched data
    df_km = df_all[df_all['year']==2024][
        ['solar_irradiance','wind_speed','solar_capacity','total_renewable_capacity']].copy()
    sc_km = StandardScaler()
    km_fit = KMeans(n_clusters=4, random_state=42, n_init=10)
    km_fit.fit(sc_km.fit_transform(df_km))
    inp_km = sc_km.transform([[solar_irradiance, wind_speed, solar_capacity, renewable_total]])
    cluster_id = km_fit.predict(inp_km)[0]
    centroids_cap = [sc_km.inverse_transform(km_fit.cluster_centers_)[i][2] for i in range(4)]
    sorted_c = sorted(range(4), key=lambda i: centroids_cap[i], reverse=True)
    cnames = {sorted_c[0]:"Solar Leaders", sorted_c[1]:"Growing States",
              sorted_c[2]:"Untapped Potential", sorted_c[3]:"Early Stage"}
    cluster_name = cnames.get(cluster_id, f"Cluster {cluster_id}")

    print(f"\n  [M4] Cluster          : {cluster_name}")

    gap_ratio = max(0, min(1, (consumption - renewable_total) / consumption))
    priority  = 'HIGH' if solar_irradiance > 140 and gap_ratio > 0.75 else \
                'MEDIUM' if solar_irradiance > 125 or gap_ratio > 0.5 else 'LOW'
    print(f"  [M2] Priority         : {priority}  (gap_ratio={gap_ratio:.2f})")

    # RF demand forecast
    df_rf = df_all.dropna(subset=['energy_consumption']).copy()
    df_rf['year_num'] = df_rf['year'] - 2020
    rf_cols = ['solar_irradiance','wind_speed','solar_capacity','wind_capacity',
               'hydro_capacity','total_renewable_capacity','year_num']
    rf_q = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    rf_q.fit(df_rf[rf_cols], df_rf['energy_consumption'])
    cons_pred = rf_q.predict([[solar_irradiance, wind_speed, solar_capacity,
                                wind_capacity, hydro_capacity, renewable_total,
                                year-2020]])[0]
    print(f"  [M1] Predicted Consumption {year}: {cons_pred:,.0f} MU")

    # Solar trend (use national avg slope)
    avg_s = np.mean([v[0] for v in state_trends.values()])
    solar_2025 = solar_capacity + avg_s * (year - 2024)
    print(f"  [M3] Predicted Solar Cap {year}  : {solar_2025:,.0f} MW")
    print(f"  [M3/M1] Demand Gap {year}        : {max(0,cons_pred-solar_2025):,.0f} MU")

    # ── Model 6: Optimization
    max_solar_add = max(target_mw * 0.6 - solar_capacity, 0)
    max_wind_add  = max(target_mw * 0.3 - wind_capacity,  0)
    max_hydro_add = max(target_mw * 0.1 - hydro_capacity, 0)
    c6 = [COST_WEIGHT*(SOLAR_COST/MAX_COST)-CO2_WEIGHT*(SOLAR_CO2/MAX_CO2_SAV),
          COST_WEIGHT*(WIND_COST/MAX_COST) -CO2_WEIGHT*(WIND_CO2/MAX_CO2_SAV),
          COST_WEIGHT*(HYDRO_COST/MAX_COST)-CO2_WEIGHT*(HYDRO_CO2/MAX_CO2_SAV)]
    A6 = [[-1,0,0],[0,-1,0],[0,0,-1],[-1,-1,-1]]
    b6 = [0, 0, 0, -supply_gap]
    bounds6 = [(0,max_solar_add),(0,max_wind_add),(0,max_hydro_add)]
    try:
        r6 = linprog(c6, A_ub=A6, b_ub=b6, bounds=bounds6, method='highs')
        if r6.success:
            sa,wa,ha = r6.x
        else:
            sa,wa,ha = supply_gap*0.6, supply_gap*0.25, supply_gap*0.15
    except:
        sa,wa,ha = supply_gap*0.6, supply_gap*0.25, supply_gap*0.15
    cost6 = sa*SOLAR_COST + wa*WIND_COST + ha*HYDRO_COST
    co2_6 = sa*SOLAR_CO2  + wa*WIND_CO2  + ha*HYDRO_CO2
    print(f"\n  [M6] Optimal Mix to Fill Gap:")
    print(f"       Solar to add : {sa:,.0f} MW")
    print(f"       Wind to add  : {wa:,.0f} MW")
    print(f"       Hydro to add : {ha:,.0f} MW")
    print(f"       Est. Cost    : ₹{cost6:,.0f} Crore")
    print(f"       CO2 Savings  : {co2_6:,.0f} tonnes/year")

    # ── Model 7: CO2 prediction
    co2_input = [[solar_capacity, wind_capacity, hydro_capacity,
                  renew_pct, plf, resource_score,
                  solar_irradiance, wind_speed, renewable_total,
                  self_suff, fossil_dep]]
    co2_pred = gb_model.predict(co2_input)[0]
    print(f"\n  [M7] Predicted CO2 Avoided   : {co2_pred:,.0f} tonnes/year")

    # ── Model 8: T&D tier
    td_input = [[solar_irradiance, wind_speed, renew_pct, resource_score,
                 cons_per_cap, plf, self_suff, fossil_dep, 0, supply_gap]]
    td_pred = le_m8.inverse_transform(rf_m8.predict(td_input))[0]
    td_msg = {'LOW':'✅ Efficient grid — ready for more renewables',
              'MEDIUM':'⚠️  Grid modernisation recommended',
              'HIGH':'🔴 Urgent grid upgrade needed before adding capacity'}
    print(f"\n  [M8] Grid Efficiency Tier    : {td_pred} — {td_msg.get(td_pred,'')}")

    # ── Model 9: Self-sufficiency forecast
    curr_ss = self_suff
    if state_name in state_trends:
        s9, i9 = state_trends[state_name]
    else:
        s9 = np.mean([v[0] for v in state_trends.values()])
        i9 = curr_ss - s9 * year
    ss_2030 = min(1.0, max(0, s9 * 2030 + i9))
    print(f"\n  [M9] Self-Sufficiency 2024   : {curr_ss:.1%}")
    print(f"  [M9] Self-Sufficiency 2030   : {ss_2030:.1%}")
    traj = 'Improving ↑' if s9 > 0.002 else 'Stagnant →' if s9 > -0.002 else 'Declining ↓'
    print(f"  [M9] Trajectory              : {traj}")

    print("═"*70)

# Demo predictions
print("\n📍 DEMO: GOA")
full_predict("Goa", solar_irradiance=156.7, wind_speed=3.85,
             solar_capacity=43.48, wind_capacity=0, hydro_capacity=0.05,
             energy_consumption=3720, population=1500000,
             plf=69.8, td_loss_pct=13.7, resource_score=73.7,
             installed_target_mw=358, year=2025)

print("\n📍 DEMO: RAJASTHAN")
full_predict("Rajasthan", solar_irradiance=127.8, wind_speed=3.09,
             solar_capacity=21347.58, wind_capacity=5193.42, hydro_capacity=23.85,
             energy_consumption=76212.9, population=80000000,
             plf=69.8, td_loss_pct=18.0, resource_score=72.0,
             installed_target_mw=40000, year=2025)

# ── Save full summary
print("\n" + "=" * 70)
print("MODEL SUMMARY")
print("=" * 70)
print(f"""
Model 6 — Optimization (LP)        : Saved → model6_optimization_results.csv
Model 7 — CO2 Predictor (GB)       : R²={train_r2_m7:.3f}, CV={cv_r2_m7.mean():.3f}
Model 8 — T&D Classifier (RF)      : Train={train_acc_m8*100:.1f}%, CV={cv_m8.mean()*100:.1f}%
Model 9 — Self-Sufficiency Forecast: Saved → model9_self_sufficiency_forecast.csv

Charts generated:
  model6_optimization.png
  model7_co2_predictor.png
  model8_grid_efficiency.png
  model9_self_sufficiency.png
""")

print("""
HOW TO EXPLAIN IN YOUR VIVA
============================
Q: What is Linear Programming (Model 6)?
A: LP finds the optimal solution to a problem with constraints.
   We defined: minimize (cost × capacity_added) subject to:
   total capacity >= demand_gap, and each source <= its maximum potential.
   scipy.optimize.linprog() solves this in microseconds per state.

Q: Why Gradient Boosting for CO2 (Model 7)?
A: Unlike Random Forest which builds trees in parallel,
   Gradient Boosting builds trees sequentially — each corrects
   the previous tree's errors. It achieves higher accuracy on
   continuous targets with complex feature interactions.

Q: What is T&D loss and why predict it (Model 8)?
A: Transmission & Distribution loss = electricity lost in the grid
   before reaching consumers. India average is ~17%. High losses mean
   even perfectly built renewable capacity is wasted. Predicting which
   states have high T&D helps prioritise grid upgrades.

Q: Why forecast self-sufficiency to 2030 (Model 9)?
A: India's National Solar Mission targets align with 2030.
   State-level self-sufficiency forecasting shows which states will
   meet their targets naturally and which need policy intervention.
   This directly supports the DWM objective of actionable insights.
""")
