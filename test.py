import pandas as pd
import os
import numpy as np
import pandas as pd
import xarray as xr

# ---------------- settings ----------------
ERA5_DIR = "./era5_local"
DATA_DIR = "."   # folder holding your 5 CSVs

FILES = {
    "ni_train": "ni_train.csv",
    "ni_val": "ni_val.csv",
    "ni_vault": "ni_TEST_VAULT.csv",
    "global_train": "global_pretrain_train.csv",
    "global_val": "global_pretrain_val.csv",
}

CHECKPOINT_SUFFIX = "_with_steering_checkpoint.csv"
FINAL_SUFFIX = "_with_steering.csv"


# ---------------- load, resuming from checkpoint if present ----------------
def load(name, path):
    ckpt = os.path.join(DATA_DIR, name + CHECKPOINT_SUFFIX)
    if os.path.exists(ckpt):
        print(f"{name}: resuming from checkpoint")
        return pd.read_csv(ckpt, parse_dates=["ISO_TIME"], low_memory=False)
    print(f"{name}: starting fresh from {path}")
    df = pd.read_csv(os.path.join(DATA_DIR, path), parse_dates=["ISO_TIME"], low_memory=False)
    df["U_STEER"] = np.nan
    df["V_STEER"] = np.nan
    return df


dfs = {name: load(name, path) for name, path in FILES.items()}


# ---------------- extract steering for one year ----------------
def extract_year(df_year, year):
    monthly = []
    for month in range(1, 13):
        fpath = os.path.join(ERA5_DIR, f"{year}_{month:02d}.nc")
        if os.path.exists(fpath):
            try:
                monthly.append(xr.open_dataset(fpath))
            except OSError as e:
                print(f"  corrupted file {fpath}, skipping: {e}")

    if not monthly:
        out = df_year.copy()
        out["U_STEER"] = np.nan
        out["V_STEER"] = np.nan
        return out

    time_dim = "valid_time" if "valid_time" in monthly[0].coords else "time"
    ds = xr.concat(monthly, dim=time_dim, join="exact") if len(monthly) > 1 else monthly[0]

    u_var = "u" if "u" in ds else "u_component_of_wind"
    v_var = "v" if "v" in ds else "v_component_of_wind"
    lev_dim = "level" if "level" in ds.dims else "pressure_level"

    u_field = ds[u_var].sel({lev_dim: [500, 700]}).mean(dim=lev_dim)
    v_field = ds[v_var].sel({lev_dim: [500, 700]}).mean(dim=lev_dim)

    out = df_year.copy()
    lon_360 = np.where(out["LON"].values < 0, out["LON"].values + 360, out["LON"].values)

    lats = xr.DataArray(out["LAT"].values, dims="points")
    lons = xr.DataArray(lon_360, dims="points")
    times = xr.DataArray(out["ISO_TIME"].values, dims="points")

    kwargs = {"latitude": lats, "longitude": lons, time_dim: times, "method": "nearest"}
    out["U_STEER"] = u_field.interp(**kwargs).values
    out["V_STEER"] = v_field.interp(**kwargs).values

    for d in monthly:
        d.close()
    return out


# ---------------- figure out which years are actually needed ----------------
all_years = sorted(set().union(*[set(df["ISO_TIME"].dt.year.unique()) for df in dfs.values()]))
already_done = sorted(dfs["ni_train"][dfs["ni_train"]["U_STEER"].notna()]["ISO_TIME"].dt.year.unique())
years_to_do = [y for y in all_years if y not in already_done]

print(f"\nTotal years needed: {len(all_years)}")
print(f"Already done: {len(already_done)}")
print(f"Remaining: {years_to_do}\n")


# ---------------- main loop ----------------
for i, year in enumerate(years_to_do):
    print(f"=== {year} ({i+1}/{len(years_to_do)}) ===")
    any_file = any(os.path.exists(os.path.join(ERA5_DIR, f"{year}_{m:02d}.nc")) for m in range(1, 13))
    if not any_file:
        print(f"  no downloaded files for {year}, marking as missing and continuing")
        continue

    for name, df in dfs.items():
        rows = df[df["ISO_TIME"].dt.year == year]
        if len(rows) == 0:
            continue
        extracted = extract_year(rows, year)
        df.loc[extracted.index, "U_STEER"] = extracted["U_STEER"].values
        df.loc[extracted.index, "V_STEER"] = extracted["V_STEER"].values
        n_valid = extracted["U_STEER"].notna().sum()
        print(f"  {name}: {n_valid}/{len(rows)} rows got real values")

    if i % 5 == 0:
        for name, df in dfs.items():
            df.to_csv(os.path.join(DATA_DIR, name + CHECKPOINT_SUFFIX), index=False)
        print("  checkpoint saved")

# ---------------- final save ----------------
for name, df in dfs.items():
    df.to_csv(os.path.join(DATA_DIR, name + FINAL_SUFFIX), index=False)

print("\nDone. Final coverage:")
for name, df in dfs.items():
    n_valid = df["U_STEER"].notna().sum()
    print(f"  {name}: {n_valid}/{len(df)} rows have steering data ({n_valid/len(df)*100:.1f}%)")

ni_train_df     = pd.read_csv('ni_train_with_steering.csv', parse_dates=['ISO_TIME'], low_memory=False)
ni_val_df       = pd.read_csv('ni_val_with_steering.csv', parse_dates=['ISO_TIME'], low_memory=False)
ni_vault_df     = pd.read_csv('ni_vault_with_steering.csv', parse_dates=['ISO_TIME'], low_memory=False)
global_train_df = pd.read_csv('global_train_with_steering.csv', parse_dates=['ISO_TIME'], low_memory=False)
global_val_df   = pd.read_csv('global_val_with_steering.csv', parse_dates=['ISO_TIME'], low_memory=False)

print("Loaded. Row counts:")
for name, df in [('ni_train', ni_train_df), ('ni_val', ni_val_df), ('ni_vault', ni_vault_df),
                  ('global_train', global_train_df), ('global_val', global_val_df)]:
    print(f"  {name}: {len(df)} rows, {df['U_STEER'].notna().sum()} with steering data")
# check the one missing NI row
missing_ni = ni_train_df[ni_train_df['U_STEER'].isna()]
print(missing_ni[['SID','ISO_TIME','LAT','LON']])

# drop NaN steering rows from global sets only (NI stays as-is, that one row is negligible)
before = len(global_train_df)
global_train_clean = global_train_df.dropna(subset=['U_STEER', 'V_STEER'])
print(f"\nglobal_train: dropped {before - len(global_train_clean)} rows ({(before-len(global_train_clean))/before*100:.1f}%), {len(global_train_clean)} remain")

before = len(global_val_df)
global_val_clean = global_val_df.dropna(subset=['U_STEER', 'V_STEER'])
print(f"global_val: dropped {before - len(global_val_clean)} rows ({(before-len(global_val_clean))/before*100:.1f}%), {len(global_val_clean)} remain")

global_train_clean.to_csv('global_train_with_steering_CLEAN.csv', index=False)
global_val_clean.to_csv('global_val_with_steering_CLEAN.csv', index=False)
ni_train_clean = ni_train_df.dropna(subset=['U_STEER', 'V_STEER'])
print(f"ni_train: dropped {len(ni_train_df) - len(ni_train_clean)} row, {len(ni_train_clean)} remain")
ni_train_clean.to_csv('ni_train_with_steering_CLEAN.csv', index=False)