import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
PATH = "trading_df.csv"
DROP_COLS = ["cred_limit", "fin_cond_index", "tax_regime"]
BASE_COLS = ["year", "vat_num", "dflt_year"]

# =========================
# LOAD RAW
# =========================
df_raw = pd.read_csv(PATH)
df_raw = df_raw.drop(columns=DROP_COLS, errors="ignore")

# Detect fin columns
fin_cols = [c for c in df_raw.columns if c not in BASE_COLS]

# All-zero (no-report) mask in RAW
all_zero_mask_raw = df_raw[fin_cols].isin([0, "0"]).all(axis=1)

# =========================
# BUILD CLEAN (as intended)
# =========================
df_clean = df_raw.copy()

# 1) remove all-zero rows (no-report years)
df_clean = df_clean[~all_zero_mask_raw].copy()

# 2) keep only until first default (inclusive)
df_clean = df_clean.sort_values(["vat_num", "year"])
df_clean["ever_defaulted"] = df_clean.groupby("vat_num")["dflt_year"].cumsum()
df_clean = df_clean[df_clean["ever_defaulted"] <= 1].copy()
df_clean = df_clean.drop(columns=["ever_defaulted"], errors="ignore")

# =========================
# CHECKS (PRINT EVERYTHING IMPORTANT)
# =========================
print("=== BASIC SHAPES ===")
print("RAW shape   :", df_raw.shape)
print("CLEAN shape :", df_clean.shape)

print("\n=== DUPLICATES CHECK ===")
dup_raw = df_raw.duplicated(["vat_num", "year"]).sum()
dup_clean = df_clean.duplicated(["vat_num", "year"]).sum()
print("RAW duplicated (vat_num,year):", dup_raw)
print("CLEAN duplicated (vat_num,year):", dup_clean)

print("\n=== ALL-ZERO (NO-REPORT) ROWS ===")
print("RAW all-zero rows count:", int(all_zero_mask_raw.sum()))
print("RAW all-zero share     :", float(all_zero_mask_raw.mean()))
all_zero_mask_clean = df_clean[fin_cols].isin([0, "0"]).all(axis=1) if len(df_clean) else pd.Series([], dtype=bool)
print("CLEAN all-zero rows count (should be 0):", int(all_zero_mask_clean.sum()) if len(df_clean) else 0)

print("\n=== TARGET DISTRIBUTIONS (ROWS) ===")
print("RAW dflt_year counts:")
print(df_raw["dflt_year"].value_counts(dropna=False).sort_index())
print("\nCLEAN dflt_year counts:")
print(df_clean["dflt_year"].value_counts(dropna=False).sort_index())

print("\n=== dflt_year VS ALL-ZERO (RAW) ===")
print("dflt_year among ALL-ZERO rows (RAW):")
print(df_raw.loc[all_zero_mask_raw, "dflt_year"].value_counts(dropna=False).sort_index())
print("\ndflt_year among NON-ZERO rows (RAW):")
print(df_raw.loc[~all_zero_mask_raw, "dflt_year"].value_counts(dropna=False).sort_index())

print("\n=== DEFAULT COMPANIES (UNIQUE vat_num) ===")
raw_def_companies = df_raw.loc[df_raw["dflt_year"] == 1, "vat_num"].nunique()
clean_def_companies = df_clean.loc[df_clean["dflt_year"] == 1, "vat_num"].nunique()
print("RAW default companies  :", raw_def_companies)
print("CLEAN default companies:", clean_def_companies)

print("\n=== MAX DEFAULTS PER COMPANY (CLEAN) ===")
max_defaults_clean = df_clean.groupby("vat_num")["dflt_year"].sum().max() if len(df_clean) else 0
print("max sum(dflt_year) per vat_num (should be <= 1):", max_defaults_clean)

print("\n=== WHERE DID DEFAULT COMPANIES GO? (RAW defaults missing in CLEAN) ===")
raw_default_set = set(df_raw.loc[df_raw["dflt_year"] == 1, "vat_num"].unique())
clean_default_set = set(df_clean.loc[df_clean["dflt_year"] == 1, "vat_num"].unique())
lost_defaults = raw_default_set - clean_default_set
print("Lost default companies count:", len(lost_defaults))

# Show a few examples
n_examples = 5
examples = list(lost_defaults)[:n_examples]
if examples:
    print(f"\n--- Examples of lost default companies (first {n_examples}) ---")
    cols_show = ["vat_num", "year", "dflt_year"]
    print("\nRAW (all rows):")
    print(df_raw[df_raw["vat_num"].isin(examples)].sort_values(["vat_num", "year"])[cols_show].head(200))

    print("\nRAW (non-zero rows only):")
    print(df_raw[(df_raw["vat_num"].isin(examples)) & (~all_zero_mask_raw)].sort_values(["vat_num", "year"])[cols_show].head(200))
else:
    print("No lost default companies (CLEAN keeps all default vat_num).")

print("\n=== LAG ANALYSIS (DEFAULT YEAR - LAST NON-DEFAULT REPORT YEAR) ===")
# default year per company in RAW (first default)
default_year = df_raw.loc[df_raw["dflt_year"] == 1].groupby("vat_num")["year"].min()

# last report year before default using dflt_year==0 rows (in RAW)
last_report = (
    df_raw[df_raw["dflt_year"] == 0]
    .groupby("vat_num")["year"]
    .max()
)

lag = (default_year - last_report).dropna()
print("Lag value counts (years):")
print(lag.value_counts().sort_index().head(50))
print("\nLag describe:")
print(lag.describe())

print("\n=== QUICK SANITY: share with lag==1, <=2, <=3 ===")
if len(lag):
    print("share lag==1 :", float((lag == 1).mean()))
    print("share lag<=2 :", float((lag <= 2).mean()))
    print("share lag<=3 :", float((lag <= 3).mean()))
else:
    print("No lag computed (no overlap between default_year and last_report).")
