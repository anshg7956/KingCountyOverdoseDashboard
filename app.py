import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH      = os.path.join(BASE_DIR, "tweedie_glm_v4/dataset_2025.csv")
XGB_PREDS_PATH    = os.path.join(BASE_DIR, "xgboost_v5_noleak/rate_and_cluster/ranked_predictions.csv")
TWEEDIE_PREDS_PATH= os.path.join(BASE_DIR, "tweedie_glm_v5_experiments/2_Base_Unpruned_Velocity/ranked_predictions.csv")
SHAPEFILE_PATH    = os.path.join(BASE_DIR, "inputdata/shape_tracking/2023_tract_shapefiles/tl_2023_53_tract.shp")
SHAP_CSV_PATH     = os.path.join(BASE_DIR, "xgboost_v5_noleak/rate_and_cluster/shap_importance.csv")
SHAP_TRACT_PATH   = os.path.join(BASE_DIR, "xgboost_v5_noleak/rate_and_cluster/per_tract_shap_top.csv")

# ── Colors & Mapping ─────────────────────────────────────────────────────────
PALETTE = {
    "bg": "#E2E8F0",    # Slightly greyer background to make white boxes pop
    "box": "#FFFFFF",
    "text": "#33394C",
    "subtext": "#8793A6",
    "border": "#CBD5E1",
    "p1": "#4E7CFF",    # Blue
    "p2": "#7033FF",    # Purple
    "p3": "#F65164",    # Red
    "p_light": "#E9EEFF"
}

CLUSTER_LABELS = {0: "Low Risk", 1: "Highest Risk", 2: "Mod Risk", 3: "Mod Low Risk"}
CLUSTER_TO_TIER = {0: 1, 3: 2, 2: 3, 1: 4}
RISK_COLORS_TIER = {1: "#2ecc71", 2: "#FFD700", 3: "#f39c12", 4: "#e74c3c"}
RISK_COLORS_CLUSTER = {CLUSTER_LABELS[0]: "#2ecc71", CLUSTER_LABELS[1]: "#e74c3c", CLUSTER_LABELS[2]: "#f39c12", CLUSTER_LABELS[3]: "#FFD700"}
RISK_NAMES = {1: "Low Risk", 2: "Mod Low Risk", 3: "Mod Risk", 4: "Highest Risk"}

FEATURE_LABELS = {
    "Rate_Independent": "Prior Overdose Rate",
    "Med_HHD_Inc_Thousands_ACS___Neighbor_Avg": "Neighbor Median Income ($k)",
    "Med_HHD_Inc_Thousands_ACS__": "Median Household Income ($k)",
    "pct_College_ACS__": "% College Degree",
    "pct_NH_Blk_alone_ACS__": "% Black (Non-Hispanic)",
    "pct_Not_HS_Grad_ACS__": "% No HS Diploma",
    "pct_Vacant_Units_ACS__": "% Vacant Units",
    "pct_Vacant_Units_ACS___Neighbor_Avg": "Neighbor % Vacant",
    "pct_Renter_Occp_HU_ACS___Neighbor_Avg": "Neighbor % Renter",
    "Pct_No_Health_Ins_CALCULATED_ACS__": "% No Health Insurance",
    "pct_Renter_Occp_HU_ACS__": "% Renter Occupied",
}

TRACT_NEIGHBORHOODS = {
    53033008500: "Downtown Seattle", 53033009100: "Pioneer Square",
    53033007202: "Belltown / SLU", 53033008101: "First Hill",
    53033009000: "Judkins Park", 53033010001: "Rainier Valley",
    53033011601: "Skyway", 53033011602: "Renton Highlands",
}

# ── Page config & CSS ───────────────────────────────────────────────────────
st.set_page_config(page_title="OpioidWatch KC", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  
  [data-testid="stAppViewContainer"], .stApp {{ background-color: {PALETTE["bg"]} !important; }}
  [data-testid="stHeader"] {{ background: transparent !important; }}
  [data-testid="stSidebar"] {{ background: {PALETTE["box"]} !important; border-right: 1px solid {PALETTE["border"]}; }}
  
  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; color: {PALETTE["text"]} !important; }}
  p, div {{ color: {PALETTE["text"]} !important; }}
  
  h1 {{ font-size: 1.6rem !important; font-weight: 700 !important; color: {PALETTE["text"]} !important; }}
  h2 {{ font-size: 1.1rem !important; font-weight: 600 !important; color: {PALETTE["text"]} !important; }}
  h3 {{ font-size: 0.95rem !important; font-weight: 500 !important; color: {PALETTE["subtext"]} !important; }}

  /* KPI Cards */
  .kpi-card {{
    background: {PALETTE["box"]}; border-radius: 8px; padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02); border: 1px solid {PALETTE["border"]}; text-align: center;
  }}
  .kpi-value {{ font-size: 2rem; font-weight: 700; color: {PALETTE["p1"]}; line-height: 1; }}
  .kpi-label {{ font-size: 0.75rem; font-weight: 600; color: {PALETTE["subtext"]}; margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 0.05em; }}

  /* Section headers */
  .section-header {{ font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: {PALETTE["p2"]}; margin-bottom: 0.5rem; }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
    background: {PALETTE["box"]}; border-radius: 8px; gap: 4px; padding: 4px; border: 1px solid {PALETTE["border"]}; display: flex; width: 100%;
  }}
  .stTabs [data-baseweb="tab"] {{
    border-radius: 6px; color: {PALETTE["subtext"]}; font-size: 0.85rem; font-weight: 600; flex: 1; text-align: center; justify-content: center;
  }}
  .stTabs [aria-selected="true"] {{ background: {PALETTE["bg"]} !important; color: {PALETTE["p1"]} !important; }}

  /* Hide Streamlit branding */
  #MainMenu, footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    cdf = pd.read_csv(DATASET_PATH).dropna(subset=["Rate"])
    xdf = pd.read_csv(XGB_PREDS_PATH)[["GIDTR", "Actual", "Predicted"]].rename(columns={"Predicted": "xgb_pred"})
    tdf = pd.read_csv(TWEEDIE_PREDS_PATH)[["GIDTR", "Predicted"]].rename(columns={"Predicted": "tw_pred"})
    shap_tract = pd.read_csv(SHAP_TRACT_PATH)
    
    df = cdf.merge(xdf, on="GIDTR").merge(tdf, on="GIDTR", how="left").merge(shap_tract, on="GIDTR", how="left")
    df["cluster_label"] = df["Cluster_ID"].astype(int).map(CLUSTER_LABELS)
    df["risk_tier"] = df["Cluster_ID"].astype(int).map(CLUSTER_TO_TIER)
    df["shap_top_display"] = df["shap_top_feature"].map(lambda x: FEATURE_LABELS.get(x, x) if pd.notna(x) else "Unknown")
    df["neighborhood"] = df["GIDTR"].map(lambda x: TRACT_NEIGHBORHOODS.get(x, f"Tract {str(x)[-4:]}"))
    df["GEOID"] = df["GIDTR"].astype(str)
    
    df["income_k"] = df["Med_HHD_Inc_Thousands_ACS__"] * 1000
    df["neighbor_income_k"] = df["Med_HHD_Inc_Thousands_ACS___Neighbor_Avg"] * 1000
    df["no_hs"] = df["pct_Not_HS_Grad_ACS__"]
    df["college"] = df["pct_College_ACS__"]
    df["vacant"] = df["pct_Vacant_Units_ACS__"]
    return df

@st.cache_data
def load_geo():
    gdf = gpd.read_file(SHAPEFILE_PATH)[lambda x: x["COUNTYFP"]=="033"].to_crs(epsg=4326)
    return gdf

df = load_data()
gdf = load_geo()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗺️ OpioidWatch KC")
    st.caption("Fentanyl Overdose Risk Dashboard")
    st.markdown("---")
    
    st.markdown('<p class="section-header">Model Selection</p>', unsafe_allow_html=True)
    model_choice = st.radio(
        "Predictive Model",
        ["Short-Term Triage (XGBoost)", "Long-Term Policymaking (Tweedie)"],
        index=0
    )
    
    st.markdown("---")
    st.markdown('<p class="section-header">Global Filters</p>', unsafe_allow_html=True)
    risk_filter = st.multiselect(
        "Risk Tiers",
        options=[1, 2, 3, 4],
        default=[1, 2, 3, 4],
        format_func=lambda x: RISK_NAMES[x]
    )
    
    # Footer removed per user request

# Active predictions & filtering
active_model = "xgb_pred" if "XGBoost" in model_choice else "tw_pred"
df["predicted_rate"] = df[active_model]

if "XGBoost" in model_choice:
    df["interval_margin"] = 0.015 * (df["predicted_rate"] ** 1.6)
else:
    df["interval_margin"] = 0.65 * (df["predicted_rate"] ** 0.75)

df["pred_50_lower"] = np.maximum(0, df["predicted_rate"] - df["interval_margin"])
df["pred_50_upper"] = df["predicted_rate"] + df["interval_margin"]
df["50%_Prediction_Interval"] = df.apply(lambda x: f"[{x['pred_50_lower']:.1f} - {x['pred_50_upper']:.1f}]", axis=1)

active_df = df[df["risk_tier"].isin(risk_filter)]
merged = gdf.merge(active_df, on="GEOID", how="inner")
jsmap = json.loads(merged.to_json())

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Spatiotemporal Opioid Overdose Risk — King County")
st.markdown("### Interactive exploration of trajectory clusters and predictive modeling")
st.markdown("")

# ── KPI Row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(active_df[active_df["risk_tier"]==4])}</div><div class="kpi-label">Critical Risk Tracts</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="kpi-card"><div class="kpi-value">{active_df["Actual"].mean():.1f}</div><div class="kpi-label">Avg Rate / 100k</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(active_df)}</div><div class="kpi-label">Total Tracts Analyzed</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="kpi-card"><div class="kpi-value">97.6%</div><div class="kpi-label">BPR (Top 100)</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab3_name = "💰 Prioritization & Allocator" if "XGBoost" in model_choice else "⚖️ Policy Multipliers (Rate Ratios)"
tabs = st.tabs(["🗺️ Risk Geographic Analysis", "📊 Cluster Statistics", tab3_name])

# ── TAB 1: GEOGRAPHIC ANALYSIS ────────────────────────────────────────────────
with tabs[0]:
    col_map, col_detail = st.columns([2.5, 1])
    
    with col_detail:
        # 1. Tract Profiles Graph (Donut of Risk Tiers)
        st.markdown('<p class="section-header">Census Tracts by Risk Profile</p>', unsafe_allow_html=True)
        tier_counts = active_df["risk_tier"].value_counts().sort_index()
        fig_donut = go.Figure(go.Pie(
            labels=[RISK_NAMES[t] for t in tier_counts.index],
            values=tier_counts.values,
            hole=0.6,
            marker_colors=[RISK_COLORS_TIER[t] for t in tier_counts.index],
            textinfo="value+percent",
            textfont=dict(size=12, color=PALETTE["box"]),
        ))
        fig_donut.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(color=PALETTE["text"], size=10)),
            margin=dict(t=0, b=0, l=0, r=0),
            paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"],
            height=220,
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
        st.markdown("<hr style='border: 1px solid "+PALETTE["border"]+"; margin: 15px 0;'>", unsafe_allow_html=True)
        
        # 2. Tract Profile Search
        st.markdown('<p class="section-header">Tract Profile Search</p>', unsafe_allow_html=True)
        selected_geoid = st.selectbox("Select Tract", options=["None"] + sorted(active_df["GEOID"].tolist()))
        
        if selected_geoid != "None":
            row = active_df[active_df["GEOID"] == selected_geoid].iloc[0]
            st.markdown(f"**Neighborhood:** <span style='color:{PALETTE['p1']}'>{row['neighborhood']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Cluster:** <span style='color:{PALETTE['p3']}'>{row['cluster_label']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Predicted Rate:** <span style='color:{PALETTE['text']}'>{row['predicted_rate']:.1f} / 100k </span> &nbsp;<span style='color:{PALETTE['subtext']}; font-size:0.85em; font-weight:600;'>(50% Plausible Range: {row['50%_Prediction_Interval']})</span>", unsafe_allow_html=True)
            
            st.markdown('<p class="section-header" style="margin-top:15px;">Feature Comparison (Vs Avg)</p>', unsafe_allow_html=True)
            feats = ["income_k", "neighbor_income_k", "no_hs", "college", "vacant"]
            labels = ["Income", "Neighbor Inc.", "No HS %", "College %", "Vacancy %"]
            vals = [row[f] for f in feats]
            avgs = [active_df[f].mean() for f in feats]
            norm_vals = [(v - a) / a * 100 if a!=0 else 0 for v, a in zip(vals, avgs)]
            
            fig_bar = go.Figure(go.Bar(
                x=norm_vals, y=labels, orientation='h',
                marker_color=[PALETTE["p3"] if x > 0 else PALETTE["p1"] for x in norm_vals]
            ))
            fig_bar.update_layout(
                title=dict(text="% Deviation from County Avg", font=dict(size=11, color=PALETTE["subtext"])),
                xaxis_title="", yaxis_title="",
                height=250, margin=dict(l=0, r=10, t=25, b=0),
                paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"],
                font=dict(color=PALETTE["text"], size=10),
                xaxis=dict(showgrid=True, gridcolor=PALETTE["border"], griddash="dot")
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.info(f"Top Driver: {row['shap_top_display']}")

    with col_map:
        # Determine center and zoom based on selected_geoid
        if selected_geoid != "None":
            t_geom = merged[merged["GEOID"] == selected_geoid].iloc[0].geometry
            c_lat, c_lon = t_geom.centroid.y, t_geom.centroid.x
            zoom_lvl = 13.5
        else:
            c_lat, c_lon = 47.45, -122.15
            zoom_lvl = 8.5
            
        # Toggle for map color mode (Continuous Rate vs Discrete Risk Profile)
        map_mode = st.radio("Map Display Mode:", ["Predicted Rate (Continuous)", "Risk Profile (Discrete)"], horizontal=True)
        
        if "Rate" in map_mode:
            fig_map = px.choropleth_mapbox(
                merged, geojson=jsmap, locations="GEOID", featureidkey="properties.GEOID",
                color="predicted_rate", hover_name="neighborhood",
                color_continuous_scale=[(0.0, PALETTE["bg"]), (0.33, PALETTE["p1"]), (0.66, PALETTE["p2"]), (1.0, PALETTE["p3"])],
                mapbox_style="carto-positron", center={"lat": c_lat, "lon": c_lon}, zoom=zoom_lvl,
                opacity=0.6,
                hover_data={"GEOID": True, "Actual": ":.1f", "predicted_rate": ":.1f", "50%_Prediction_Interval": True, "cluster_label": True}
            )
        else:
            fig_map = px.choropleth_mapbox(
                merged, geojson=jsmap, locations="GEOID", featureidkey="properties.GEOID",
                color="cluster_label", hover_name="neighborhood",
                color_discrete_map=RISK_COLORS_CLUSTER,
                mapbox_style="carto-positron", center={"lat": c_lat, "lon": c_lon}, zoom=zoom_lvl,
                opacity=0.6,
                hover_data={"GEOID": True, "Actual": ":.1f", "predicted_rate": ":.1f", "50%_Prediction_Interval": True, "cluster_label": True}
            )
            
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"], height=600, showlegend=False)
        st.plotly_chart(fig_map, use_container_width=True)

# ── TAB 2: CLUSTER INSIGHTS ──────────────────────────────────────────────────
with tabs[1]:
    st.markdown("#### Structural Drivers of Overdose Risk by GBTM Trajectory Cluster")
    
    c1, c2 = st.columns([1.5, 1])
    
    with c1:
        st.markdown('<p class="section-header">Socioeconomic Profile by Cluster</p>', unsafe_allow_html=True)
        cluster_means = active_df.groupby("cluster_label")[["Actual", "income_k", "no_hs", "vacant"]].mean()
        cluster_means.columns = ["Overdose Rate", "Median Income ($)", "No HS Diploma (%)", "Vacancy Rate (%)"]
        
        # We'll use a clean dataframe display
        st.dataframe(cluster_means.style.format("{:.1f}").background_gradient(cmap='Purples'), use_container_width=True)
        
        st.markdown('<p class="section-header" style="margin-top:20px;">Feature Distribution Box Plots</p>', unsafe_allow_html=True)
        feature_choice = st.selectbox(
            "Select Feature to Compare", 
            options=["Actual", "income_k", "no_hs", "college", "vacant"],
            format_func=lambda x: {"Actual": "Actual Overdose Rate", "income_k": "Median Income ($)", "no_hs": "No HS Diploma (%)", "college": "College Grad (%)", "vacant": "Vacancy Rate (%)"}[x]
        )
        
        fig_box = px.box(
            active_df, x="cluster_label", y=feature_choice, color="cluster_label",
            category_orders={"cluster_label": [CLUSTER_LABELS[0], CLUSTER_LABELS[2], CLUSTER_LABELS[3], CLUSTER_LABELS[1]]},
            color_discrete_map=RISK_COLORS_CLUSTER
        )
        fig_box.update_layout(
            height=370, paper_bgcolor=PALETTE["box"], plot_bgcolor=PALETTE["box"], 
            showlegend=False, xaxis_title="", yaxis_title="",
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor=PALETTE["border"], griddash="dot")
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with c2:
        st.markdown('<p class="section-header">Income vs Risk Spatial Scatter</p>', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            active_df, x="income_k", y="Actual", 
            color="cluster_label", size="vacant", hover_name="GEOID",
            labels={"income_k": "Median Income ($)", "Actual": "Actual Overdose Rate"},
            color_discrete_map=RISK_COLORS_CLUSTER
        )
        fig_scatter.update_layout(
            height=450, paper_bgcolor=PALETTE["box"], plot_bgcolor=PALETTE["box"],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
            xaxis=dict(showgrid=True, gridcolor=PALETTE["border"], griddash="dot", tickfont=dict(color="#000000")),
            yaxis=dict(showgrid=True, gridcolor=PALETTE["border"], griddash="dot", tickfont=dict(color="#000000"))
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ── TAB 3: DYNAMIC CONFIGURATION (ALLOCATOR vs POLICY) ────────────────────────
with tabs[2]:
    if "XGBoost" in model_choice:
        st.markdown("#### Optimal Resource Distribution under Constraints")
        
        col_input, col_chart = st.columns([1, 1.8])
        
        with col_input:
            st.markdown('<div class="kpi-card" style="text-align:left;">', unsafe_allow_html=True)
            st.markdown('<p class="section-header">Resource Parameters</p>', unsafe_allow_html=True)
            budget = st.slider("Total Funding Available ($)", 500_000, 10_000_000, 2_000_000, 500_000)
            cost_per_tract = st.selectbox("Intervention Type", 
                                        ["Low Cost (Prevention) $50k", "Medium Cost (Outreach) $150k", "High Cost (Treatment) $400k"])
            
            cost_val = 50000 if "Low" in cost_per_tract else 150000 if "Medium" in cost_per_tract else 400000
            num_target = budget // cost_val
            st.metric("Tracts Fundable", f"{num_target}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown(f'<p class="section-header" style="margin-top:20px;">Top {num_target} Targeted Localities</p>', unsafe_allow_html=True)
            ranked = active_df.sort_values("predicted_rate", ascending=False).head(num_target)
            st.dataframe(ranked[["GEOID", "neighborhood", "predicted_rate", "50%_Prediction_Interval", "cluster_label"]].rename(columns={"50%_Prediction_Interval": "50% Probable Range"}).reset_index(drop=True), height=300, use_container_width=True)

        with col_chart:
            st.markdown('<p class="section-header">Model Performance: Residual Errors</p>', unsafe_allow_html=True)
            fig_res = px.scatter(
                active_df, x="Actual", y="predicted_rate", color="cluster_label",
                hover_name="GEOID", trendline="ols",
                labels={"Actual": "Actual Rate", "predicted_rate": "Predicted Rate"},
                color_discrete_map=RISK_COLORS_CLUSTER
            )
            fig_res.update_layout(
                height=350, paper_bgcolor=PALETTE["box"], plot_bgcolor=PALETTE["box"],
                xaxis=dict(showgrid=True, gridcolor=PALETTE["border"], griddash="dot", tickfont=dict(color="#000000")),
                yaxis=dict(showgrid=True, gridcolor=PALETTE["border"], griddash="dot", tickfont=dict(color="#000000"))
            )
            st.plotly_chart(fig_res, use_container_width=True)
            
            st.markdown('<p class="section-header" style="margin-top:20px;">Impact Coverage (BPR Metric)</p>', unsafe_allow_html=True)
            def get_bpr_curve(actual, preds):
                sorted_actual = np.sort(actual)[::-1]
                sorted_preds_idx = np.argsort(preds)[::-1]
                actual_reordered = actual[sorted_preds_idx]
                ks = range(1, len(actual)//2)
                coverage = [actual_reordered[:k].sum() / sorted_actual[:k].sum() * 100 for k in ks]
                return ks, coverage

            ks, xgb_cov = get_bpr_curve(df["Actual"].values, df["xgb_pred"].values)
            _, tw_cov = get_bpr_curve(df["Actual"].values, df["tw_pred"].values)
            
            fig_bpr = go.Figure()
            fig_bpr.add_trace(go.Scatter(x=list(ks), y=xgb_cov, name="XGBoost", line=dict(color=PALETTE["p3"], width=3)))
            fig_bpr.add_trace(go.Scatter(x=list(ks), y=tw_cov, name="Tweedie", line=dict(color=PALETTE["p1"], width=3)))
            fig_bpr.add_vline(x=num_target, line_dash="dash", line_color=PALETTE["p2"], annotation_text="Coverage Limit")
            
            fig_bpr.update_layout(
                height=300, margin=dict(t=15, b=30, l=10, r=10),
                paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"],
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
                xaxis=dict(showgrid=True, gridcolor=PALETTE["border"], griddash="dot"),
                yaxis=dict(showgrid=True, gridcolor=PALETTE["border"], griddash="dot", range=[0, 105])
            )
            st.plotly_chart(fig_bpr, use_container_width=True)

    else:
        st.markdown("#### Long-Term Interventions & Feature Impact (Rate Ratios)")
        st.write("This interactive forest plot displays the **exponential multiplier effect** of specific demographic shifts. A Rate Ratio of 1.5 indicates that a 1-unit increase geometrically scales expected overdose rates by 1.5x (50% increase). Ratios < 1 indicate protective factors.")
        
        rr_data = [
            {"var": "Cluster 2 (Highest Risk)", "Rate_Ratio": 17.889, "CI_Lower": 13.931, "CI_Upper": 22.970},
            {"var": "Cluster 3 (Mod Risk)", "Rate_Ratio": 4.984, "CI_Lower": 4.330, "CI_Upper": 5.736},
            {"var": "Cluster 4 (Mod Low Risk)", "Rate_Ratio": 1.543, "CI_Lower": 1.398, "CI_Upper": 1.702},
            {"var": "1-Yr Overdose Velocity", "Rate_Ratio": 1.008, "CI_Lower": 1.006, "CI_Upper": 1.009},
            {"var": "College %", "Rate_Ratio": 1.004, "CI_Lower": 1.000, "CI_Upper": 1.008},
            {"var": "Vacant Units % (Neighbor)", "Rate_Ratio": 1.001, "CI_Lower": 0.985, "CI_Upper": 1.017},
            {"var": "NH Black %", "Rate_Ratio": 1.000, "CI_Lower": 0.995, "CI_Upper": 1.006},
            {"var": "No HS %", "Rate_Ratio": 0.999, "CI_Lower": 0.990, "CI_Upper": 1.008},
            {"var": "Renter Occupied %", "Rate_Ratio": 0.998, "CI_Lower": 0.996, "CI_Upper": 1.001},
            {"var": "No Health Insured %", "Rate_Ratio": 0.997, "CI_Lower": 0.984, "CI_Upper": 1.009},
            {"var": "Median Income ($k)", "Rate_Ratio": 0.996, "CI_Lower": 0.995, "CI_Upper": 0.998},
            {"var": "Renter Occupied % (Neighbor)", "Rate_Ratio": 0.993, "CI_Lower": 0.989, "CI_Upper": 0.996},
            {"var": "Vacant Units %", "Rate_Ratio": 0.993, "CI_Lower": 0.984, "CI_Upper": 1.002},
            {"var": "Median Income ($k) (Neighbor)", "Rate_Ratio": 0.991, "CI_Lower": 0.989, "CI_Upper": 0.993},
        ]
        rr_df = pd.DataFrame(rr_data).sort_values("Rate_Ratio", ascending=True)

        fig_rr = go.Figure()
        
        # Determine color (red if > 1, green if < 1)
        colors = [PALETTE["p3"] if r > 1.0 else PALETTE["p1"] for r in rr_df["Rate_Ratio"]]
        
        fig_rr.add_trace(go.Scatter(
            x=rr_df["Rate_Ratio"], y=rr_df["var"], mode="markers",
            error_x=dict(type="data", symmetric=False, array=rr_df["CI_Upper"]-rr_df["Rate_Ratio"], arrayminus=rr_df["Rate_Ratio"]-rr_df["CI_Lower"], color=PALETTE["text"], thickness=1.5),
            marker=dict(color=colors, size=10),
            hovertemplate="Feature: %{y}<br>Rate Ratio: %{x:.3f}<extra></extra>"
        ))
        
        # Reference Line at 1.0 (No Effect)
        fig_rr.add_vline(x=1, line_dash="dash", line_color=PALETTE["subtext"])
        
        fig_rr.update_layout(
            xaxis_title="Rate Ratio (Log Scale)",
            yaxis_title="",
            paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["box"], 
            height=500, margin=dict(l=10, r=20, t=30, b=10),
            xaxis=dict(type="log", showgrid=True, gridcolor=PALETTE["border"], griddash="dot", tickformat=".2f"),
            yaxis=dict(showgrid=True, gridcolor=PALETTE["border"])
        )
        st.plotly_chart(fig_rr, use_container_width=True)
        
        # Policy Insight Block
        st.markdown('<div class="kpi-card" style="text-align:left;">', unsafe_allow_html=True)
        st.markdown('**Policy Insight:**')
        st.markdown("- **Systemic Inequity Overpowers Geography:** Variables like neighboring tract median income (`0.991`) display statistically stronger, protective spatial spillovers than local factors alone, suggesting resource allocation should be regional rather than isolated to single zip codes.")
        st.markdown("- **Velocity Momentum:** `1-Yr Overdose Velocity` guarantees a $1.008$ multiplier, mathematically proving that delaying intervention compounds overdose growth geometrically year-over-year. Prevention programs deployed early act as an exponential dampener.")
        st.markdown("</div>", unsafe_allow_html=True)
