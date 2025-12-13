# Feature engineering
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 1) CONFIG: data load + output paths
# ---------------------------------------------------------------------
DATASET_PATH = "data/dataset.csv"
OUTPUT_PATH_FULL = "./data/Feature_Engineering/dataset_enriched.csv"
OUTPUT_PATH_SELECTED = "./data/Feature_Engineering/dataset_engineered.csv"  # new + untouched + phase

# ---------------------------------------------------------------------
# 2) LOAD DATA
# ---------------------------------------------------------------------
df = pd.read_csv(DATASET_PATH)
original_cols = df.columns.tolist()

# ---------------------------------------------------------------------
# 3) PROCESSING PIPELINE: create engineered features
# ---------------------------------------------------------------------
suffixes = ("minimo", "maximo", "soma", "media", "desvio")
pattern = re.compile(rf"^(?P<prop>.+)_(?P<suf>{'|'.join(suffixes)})$")

# Map: base property -> available suffix columns
prop_map = {}
for c in original_cols:
    m = pattern.match(c)
    if m:
        base = m.group("prop")
        suf = m.group("suf")
        prop_map.setdefault(base, {})[suf] = c

def safe_divide(numer: pd.Series, denom: pd.Series) -> np.ndarray:
    numer = numer.astype(float)
    denom = denom.astype(float)
    return np.where(denom == 0, np.nan, numer / denom)

new_series = {}          # name -> pd.Series/np.ndarray
new_names = []           # actual names added to df (collision-safe)
used_input_cols = set()  # original columns used to generate any new features

# 3.1 Range, CoV, Min/Max ratio
for base, parts in prop_map.items():
    if "maximo" in parts and "minimo" in parts:
        nm = f"{base}_range"
        new_series[nm] = df[parts["maximo"]] - df[parts["minimo"]]
        used_input_cols.update([parts["maximo"], parts["minimo"]])

    if "desvio" in parts and "media" in parts:
        nm = f"{base}_cov"
        new_series[nm] = safe_divide(df[parts["desvio"]], df[parts["media"]])
        used_input_cols.update([parts["desvio"], parts["media"]])

    if "minimo" in parts and "maximo" in parts:
        nm = f"{base}_min_to_max_ratio"
        new_series[nm] = safe_divide(df[parts["minimo"]], df[parts["maximo"]])
        used_input_cols.update([parts["minimo"], parts["maximo"]])

# 3.2 Binary indicators
for base, parts in prop_map.items():
    if "desvio" in parts:
        col = parts["desvio"]
        nm = f"has_zero_{base}_desvio"
        new_series[nm] = (df[col] == 0).astype(int)
        used_input_cols.add(col)

    if "minimo" in parts:
        col = parts["minimo"]
        nm = f"has_zero_{base}_minimo"
        new_series[nm] = (df[col] == 0).astype(int)
        used_input_cols.add(col)

# 3.3 Synthesis interaction features (only if both columns exist)
def add_interaction(feature_candidates, syn_group_name, out_name):
    base_col = next((c for c in feature_candidates if c in df.columns), None)
    if base_col is not None and syn_group_name in df.columns:
        return out_name, df[base_col] * df[syn_group_name], base_col, syn_group_name
    return None

interactions_specs = [
    (["entalpia-oxidos_media", "entalpia_oxidos_media"], "syn_group_combustion", "entalpia_oxidos_media_x_combustion"),
    (["gibbs-oxidos_minimo", "gibbs_oxidos_minimo"], "syn_group_hydrothermal", "gibbs_oxidos_minimo_x_hydrothermal"),
    (["melting_point_media"], "syn_group_solid_state", "melting_point_media_x_solid_state"),
    (["boiling_point_media"], "syn_group_solvo", "boiling_point_media_x_solvo"),
    (["atomic_en_paul_media"], "syn_group_chemical_co_precipitation", "atomic_en_paul_media_x_chemical_co_precipitation"),
    (["atomic_radius_desvio"], "syn_group_ball_mill", "atomic_radius_desvio_x_ball_mill"),
    (["VEC_media"], "syn_group_hydrothermal", "VEC_media_x_hydrothermal"),
    (["density_of_solid_desvio"], "syn_group_mechanochemical", "density_of_solid_desvio_x_mechanochemical"),
    (["thermal_conduct_media"], "syn_group_combustion", "thermal_conduct_media_x_combustion"),
]

for feature_candidates, syn_name, out_name in interactions_specs:
    res = add_interaction(feature_candidates, syn_name, out_name)
    if res is not None:
        k, v, base_col, syn_col = res
        new_series[k] = v
        used_input_cols.update([base_col, syn_col])

# 3.4 Merge new features (avoid overwriting)
for k, v in new_series.items():
    final_name = k if k not in df.columns else f"{k}__new"
    df[final_name] = v
    new_names.append(final_name)

# ---------------------------------------------------------------------
# 4) SAVE: full enriched df + engineered (new + untouched + phase)
# ---------------------------------------------------------------------
if "phase" not in df.columns:
    raise ValueError('Target column "phase" not found in the dataset.')

df.to_csv(OUTPUT_PATH_FULL, index=False)

used_input_cols.discard("phase")
untouched_originals = sorted(list(set(original_cols) - used_input_cols))

selected_cols = untouched_originals + new_names
selected_cols = [c for c in selected_cols if c != "phase"] + ["phase"]

df_selected = df[selected_cols].copy()
df_selected.to_csv(OUTPUT_PATH_SELECTED, index=False)
