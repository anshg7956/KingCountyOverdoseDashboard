# OpioidWatch KC — Setup & Deployment Guide

## Local Setup (run on your machine)

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py
```

---

## Connecting Your Real Data

Open `app.py` and find the two functions marked with ╔══╗ banners:

### 1. `load_data()` — Replace with your real tract predictions
Your DataFrame needs these columns:
| Column | Description |
|---|---|
| `GEOID` | Census tract ID — must match your shapefile |
| `predicted_rate` | Model output: overdose rate per 100k |
| `risk_tier` | Integer 1–4 (1=Low, 4=Critical) |
| `cluster_label` | GBTM cluster name string |
| `shap_top_feature` | Top SHAP driver for that tract |
| `neighbor_median_income` | Raw feature value |
| `neighbor_pct_vacancy` | Raw feature value |
| `pct_no_hs_diploma` | Raw feature value |
| `neighbor_pct_renter` | Raw feature value |
| `pct_renter_occupancy` | Raw feature value |
| `pct_vacant_units` | Raw feature value |
| `pct_black` | Raw feature value |
| `median_income` | Raw feature value |
| `pct_college` | Raw feature value |
| `pct_no_health_ins` | Raw feature value |

Example replacement:
```python
def load_data():
    df = pd.read_csv("your_predictions.csv")
    return df
```

### 2. `load_shapefile()` — Point to your .shp file
```python
def load_shapefile():
    gdf = gpd.read_file("path/to/king_county_tracts.shp")
    gdf = gdf.to_crs(epsg=4326)
    # Make sure GEOID column exists and matches tract_df
    return gdf
```

### 3. `SHAP_WEIGHTS` — Replace with your actual SHAP values
Update the dictionary with your real global SHAP importances from your model.

---

## Public Deployment (get a shareable URL for judges)

### Option A: Streamlit Community Cloud (free, easiest)
1. Push your code to a GitHub repo (keep your data files small or synthetic for public repos)
2. Go to share.streamlit.io
3. Connect your repo → deploy
4. You get a public URL like `https://yourname-opioidswatch.streamlit.app`

### Option B: Keep it local, show on your laptop
Just run `streamlit run app.py` and demo live. No deployment needed for science fair.

---

## What to say to judges about the dashboard

> "Rather than leaving our findings in a research paper, we built a deployment-ready 
> policy tool. A King County health official could open this today, enter their budget, 
> and receive a ranked list of census tracts to prioritize — along with projections of 
> how many overdose cases specific interventions would prevent."

That framing — *from research to deployable tool* — is what separates finalist projects.
