import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
PATH = "trading_df.csv"
DROP_COLS = ["cred_limit", "fin_cond_index", "tax_regime"]
BASE_COLS = ["year", "vat_num", "dflt_year"]  # dflt_year == target: default in (year+1)

# =========================
# LOAD + BASIC PREP
# =========================
df_raw = pd.read_csv(PATH)
df_raw = df_raw.drop(columns=DROP_COLS, errors="ignore")

# Columns with financials/features (everything except base)
fin_cols = [c for c in df_raw.columns if c not in BASE_COLS]

# Sort once (important for many checks)
df_raw = df_raw.sort_values(["vat_num", "year"]).reset_index(drop=True)

# =========================
# HELPER MASKS
# =========================
# "No report" year (all features are 0 / '0') in RAW
all_zero_mask = df_raw[fin_cols].isin([0, "0"]).all(axis=1)

# =========================
# 1) BASIC SANITY CHECKS
# =========================
print("=== 1) BASIC SHAPE / DUPLICATES ===")
print("RAW shape:", df_raw.shape)
dup = df_raw.duplicated(["vat_num", "year"]).sum()
print("Duplicated (vat_num, year):", dup)

print("\n=== 2) TARGET (dflt_year) SANITY ===")
print("Unique values of dflt_year:", sorted(df_raw["dflt_year"].dropna().unique().tolist()))
print("dflt_year value counts:")
print(df_raw["dflt_year"].value_counts(dropna=False).sort_index())

# Check that dflt_year is only 0/1 (or NaN)
bad_target = df_raw.loc[~df_raw["dflt_year"].isin([0, 1]) & df_raw["dflt_year"].notna(), ["vat_num", "year", "dflt_year"]]
print("\nRows with non {0,1} dflt_year:", len(bad_target))
if len(bad_target):
    print(bad_target.head(20))

# =========================
# 2) REPORTING / ZERO-ROWS DIAGNOSTICS
# =========================
print("\n=== 3) NO-REPORT (ALL-ZERO) DIAGNOSTICS ===")
print("All-zero rows count:", int(all_zero_mask.sum()))
print("All-zero rows share:", float(all_zero_mask.mean()))

print("\nTarget distribution among all-zero rows:")
print(df_raw.loc[all_zero_mask, "dflt_year"].value_counts(dropna=False).sort_index())

print("\nTarget distribution among non-zero rows:")
print(df_raw.loc[~all_zero_mask, "dflt_year"].value_counts(dropna=False).sort_index())

# =========================
# 3) COMPANY-LEVEL EVENT CONSISTENCY (dflt_year == default in next year)
# =========================
print("\n=== 4) COMPANY-LEVEL EVENT CONSISTENCY ===")
# First year t when target==1 (meaning default event is in t+1)
first_pos_t = df_raw.loc[df_raw["dflt_year"] == 1].groupby("vat_num")["year"].min()

print("Companies with at least one positive target:", int(first_pos_t.shape[0]))

# How many companies have multiple positives (potentially multiple events or label propagation)
pos_counts = df_raw.groupby("vat_num")["dflt_year"].sum()
n_multi_pos = int((pos_counts > 1).sum())
print("Companies with >1 positive targets:", n_multi_pos)
if n_multi_pos:
    print("Example vat_num with >1 positives:", pos_counts[pos_counts > 1].head(10))

# Define implied event year: event_year = t+1 for first positive
event_year = (first_pos_t + 1).rename("event_year")

# =========================
# 4) POST-EVENT ROWS CHECK
# =========================
print("\n=== 5) POST-EVENT ROWS CHECK (should be 0 after trimming) ===")
df_ev = df_raw.join(first_pos_t.rename("first_pos_t"), on="vat_num")
df_ev["event_year"] = df_ev["first_pos_t"] + 1

# Rows that occur on/after the implied event year for companies that have a positive
post_event_rows = df_ev[df_ev["first_pos_t"].notna() & (df_ev["year"] >= df_ev["event_year"])]

print("Rows on/after event year (RAW):", len(post_event_rows))
if len(post_event_rows):
    print("Sample post-event rows (RAW):")
    print(post_event_rows[["vat_num", "year", "dflt_year", "first_pos_t", "event_year"]].head(30))

# Also check how many companies have ANY rows after event year
post_event_companies = post_event_rows["vat_num"].nunique()
print("Companies with any post-event rows (RAW):", int(post_event_companies))

# =========================
# 5) GAP / MISSING PRE-EVENT YEAR CHECKS (t-1 availability etc.)
# =========================
print("\n=== 6) PRE-EVENT COVERAGE CHECKS ===")
# For each defaulting company, do we have a row at t = (event_year - 1)? should be True by definition of first_pos_t.
# But we can check if that row has real report (non-zero).
df_def = pd.DataFrame({"first_pos_t": first_pos_t}).join(event_year, how="left")
df_def["has_report_at_t"] = False

# mark if the row (vat_num, first_pos_t) is non-zero-report
key = df_raw.set_index(["vat_num", "year"])
# In case of missing keys, handle safely:
idx = list(zip(df_def.index, df_def["first_pos_t"].astype(int)))
exists = []
nonzero = []
for vat, y in idx:
    if (vat, y) in key.index:
        exists.append(True)
        nonzero.append(not bool(all_zero_mask.loc[key.index.get_loc((vat, y))]))
    else:
        exists.append(False)
        nonzero.append(False)

df_def["row_exists_at_t"] = exists
df_def["has_report_at_t"] = nonzero

print("Defaulting companies where row at t exists:", float(df_def["row_exists_at_t"].mean()))
print("Defaulting companies where row at t has NON-ZERO report:", float(df_def["has_report_at_t"].mean()))
print("Count missing row at t (unexpected):", int((~df_def["row_exists_at_t"]).sum()))
print("Count row at t but ALL-ZERO report:", int((df_def["row_exists_at_t"] & ~df_def["has_report_at_t"]).sum()))

# Optional: distribution of time gaps between consecutive years per company (data completeness)
print("\n=== 7) YEAR GAP DISTRIBUTION (RAW) ===")
next_year = df_raw.groupby("vat_num")["year"].shift(-1)
gaps = (next_year - df_raw["year"]).dropna()
print("Gap value counts (top 20):")
print(gaps.value_counts().sort_index().head(20))
print("Any gaps > 1:", bool((gaps > 1).any()))

# =========================
# 6) RECOMMENDED "CLEAN" VERSION (NO-REPORT REMOVED + TRIM AFTER FIRST POSITIVE)
# =========================
print("\n=== 8) BUILD CLEAN (recommended) AND RE-CHECK POST-EVENT ===")
df_clean = df_raw[~all_zero_mask].copy()
df_clean = df_clean.sort_values(["vat_num", "year"]).reset_index(drop=True)

df_clean["seen_pos"] = df_clean.groupby("vat_num")["dflt_year"].cumsum()
df_clean = df_clean[df_clean["seen_pos"] <= 1].copy()
df_clean = df_clean.drop(columns=["seen_pos"], errors="ignore")

print("CLEAN shape:", df_clean.shape)
print("CLEAN dflt_year counts:")
print(df_clean["dflt_year"].value_counts(dropna=False).sort_index())

# Post-event check on CLEAN
first_pos_t_clean = df_clean.loc[df_clean["dflt_year"] == 1].groupby("vat_num")["year"].min()
df_clean_ev = df_clean.join(first_pos_t_clean.rename("first_pos_t"), on="vat_num")
df_clean_ev["event_year"] = df_clean_ev["first_pos_t"] + 1
post_event_rows_clean = df_clean_ev[df_clean_ev["first_pos_t"].notna() & (df_clean_ev["year"] >= df_clean_ev["event_year"])]

print("\nRows on/after event year (CLEAN) [should be 0]:", len(post_event_rows_clean))
if len(post_event_rows_clean):
    print(post_event_rows_clean[["vat_num", "year", "dflt_year", "first_pos_t", "event_year"]].head(30))

# Multiple positives on CLEAN
pos_counts_clean = df_clean.groupby("vat_num")["dflt_year"].sum()
print("Companies with >1 positives (CLEAN) [should be 0]:", int((pos_counts_clean > 1).sum()))
