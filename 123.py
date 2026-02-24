import pandas as pd
import numpy as np

# =========================
# 0) LOAD
# =========================
old_path = "trading_df_old.csv"   # старый датасет
new_path = "trading_df_new.csv"   # новый датасет

df_old = pd.read_csv(old_path, low_memory=False)
df_new = pd.read_csv(new_path, low_memory=False)

ID_COLS = ["vat_num", "year"]
TARGET_COL = "dflt_year"

print("OLD shape:", df_old.shape)
print("NEW shape:", df_new.shape)

# =========================
# 1) COLUMNS / SCHEMA DIFF
# =========================
old_cols = set(df_old.columns)
new_cols = set(df_new.columns)

only_old_cols = sorted(old_cols - new_cols)
only_new_cols = sorted(new_cols - old_cols)
common_cols = sorted(old_cols & new_cols)

print("\n=== 1) COLUMNS ===")
print("Columns only in OLD:", len(only_old_cols))
print(only_old_cols[:50], "..." if len(only_old_cols) > 50 else "")
print("Columns only in NEW:", len(only_new_cols))
print(only_new_cols[:50], "..." if len(only_new_cols) > 50 else "")
print("Common columns:", len(common_cols))

# dtypes differences on common columns
dtype_diff = []
for c in common_cols:
    dt_old = str(df_old[c].dtype)
    dt_new = str(df_new[c].dtype)
    if dt_old != dt_new:
        dtype_diff.append((c, dt_old, dt_new))

print("\nDtype differences on common columns:", len(dtype_diff))
for row in dtype_diff[:30]:
    print(row)

# =========================
# 2) BASIC KEY CHECKS
# =========================
print("\n=== 2) KEY CHECKS ===")
for name, df in [("OLD", df_old), ("NEW", df_new)]:
    dup = df.duplicated(ID_COLS).sum()
    print(f"{name}: duplicated (vat_num, year) = {dup}")
    print(f"{name}: unique companies = {df['vat_num'].nunique()}")
    print(f"{name}: years = [{df['year'].min()}..{df['year'].max()}], n_years={df['year'].nunique()}")

# sets of keys
old_keys = set(map(tuple, df_old[ID_COLS].drop_duplicates().values))
new_keys = set(map(tuple, df_new[ID_COLS].drop_duplicates().values))

only_old_keys = old_keys - new_keys
only_new_keys = new_keys - old_keys
common_keys = old_keys & new_keys

print("\nKey coverage:")
print("Only in OLD keys:", len(only_old_keys))
print("Only in NEW keys:", len(only_new_keys))
print("Common keys:", len(common_keys))

# =========================
# 3) YEAR DISTRIBUTION COMPARE
# =========================
print("\n=== 3) YEAR DISTRIBUTION ===")
year_cmp = (
    pd.DataFrame({
        "old_rows": df_old["year"].value_counts(),
        "new_rows": df_new["year"].value_counts(),
    })
    .fillna(0)
    .astype(int)
    .sort_index()
)
year_cmp["delta"] = year_cmp["new_rows"] - year_cmp["old_rows"]
year_cmp["delta_pct_vs_old"] = np.where(
    year_cmp["old_rows"] > 0,
    100 * year_cmp["delta"] / year_cmp["old_rows"],
    np.nan
).round(2)

print(year_cmp)

# =========================
# 4) COMPANY COVERAGE COMPARE
# =========================
print("\n=== 4) COMPANY COVERAGE ===")
old_companies = set(df_old["vat_num"].unique())
new_companies = set(df_new["vat_num"].unique())

print("Companies only in OLD:", len(old_companies - new_companies))
print("Companies only in NEW:", len(new_companies - old_companies))
print("Companies common:", len(old_companies & new_companies))

# rows per company stats
old_len = df_old.groupby("vat_num").size().rename("old_nrows")
new_len = df_new.groupby("vat_num").size().rename("new_nrows")
comp_len_cmp = pd.concat([old_len, new_len], axis=1).fillna(0).astype(int)
comp_len_cmp["delta"] = comp_len_cmp["new_nrows"] - comp_len_cmp["old_nrows"]

print("\nRows per company stats:")
print(comp_len_cmp[["old_nrows", "new_nrows", "delta"]].describe())

print("\nTop companies with largest row loss:")
print(comp_len_cmp.sort_values("delta").head(20))

# =========================
# 5) ALL-ZERO ROWS COMPARE (on common financial cols)
# =========================
print("\n=== 5) ALL-ZERO ROWS (COMMON FEATURE SPACE) ===")

base_cols = [c for c in [*ID_COLS, TARGET_COL] if c in common_cols]
common_feature_cols = [c for c in common_cols if c not in base_cols]

def all_zero_mask(df, feature_cols):
    # считаем all-zero по общим фичам, чтобы сравнение было честным
    if len(feature_cols) == 0:
        return pd.Series(False, index=df.index)
    return df[feature_cols].isin([0, "0"]).all(axis=1)

old_all_zero = all_zero_mask(df_old, common_feature_cols)
new_all_zero = all_zero_mask(df_new, common_feature_cols)

def summarize_zero(df, mask, name):
    print(f"\n{name}:")
    print("Rows:", len(df))
    print("All-zero rows:", int(mask.sum()), f"({mask.mean():.2%})")
    if TARGET_COL in df.columns:
        print("Target in all-zero rows:")
        print(df.loc[mask, TARGET_COL].value_counts(dropna=False).sort_index())
        print("Target in non-zero rows:")
        print(df.loc[~mask, TARGET_COL].value_counts(dropna=False).sort_index())

summarize_zero(df_old, old_all_zero, "OLD")
summarize_zero(df_new, new_all_zero, "NEW")

# by year all-zero compare
old_zero_year = df_old.assign(all_zero=old_all_zero).groupby("year")["all_zero"].mean().rename("old_all_zero_share")
new_zero_year = df_new.assign(all_zero=new_all_zero).groupby("year")["all_zero"].mean().rename("new_all_zero_share")
zero_year_cmp = pd.concat([old_zero_year, new_zero_year], axis=1).sort_index()
print("\nAll-zero share by year:")
print((100 * zero_year_cmp).round(2))

# =========================
# 6) COMPARE VALUES ON COMMON KEYS + COMMON COLS
# =========================
print("\n=== 6) VALUE DIFFS ON COMMON KEYS ===")

# Оставим только common keys
old_common = df_old[df_old[ID_COLS].apply(tuple, axis=1).isin(common_keys)].copy()
new_common = df_new[df_new[ID_COLS].apply(tuple, axis=1).isin(common_keys)].copy()

# если вдруг есть дубли — возьмем первый, но лучше сначала починить источник
old_common = old_common.drop_duplicates(ID_COLS).set_index(ID_COLS)
new_common = new_common.drop_duplicates(ID_COLS).set_index(ID_COLS)

# сравнивать значения только по общим колонкам (без id)
cmp_cols = [c for c in common_cols if c not in ID_COLS]

# align
old_common = old_common.sort_index()
new_common = new_common.sort_index()
old_common, new_common = old_common.align(new_common, join="inner", axis=0)

# normalize "0"/0 for object columns before compare (по желанию)
def normalize_for_compare(s):
    if s.dtype == "object":
        return s.replace("0", 0)
    return s

diff_stats = []
for c in cmp_cols:
    s_old = normalize_for_compare(old_common[c])
    s_new = normalize_for_compare(new_common[c])

    # считаем NaN == NaN как совпадение
    eq = (s_old == s_new) | (s_old.isna() & s_new.isna())
    diff_cnt = int((~eq).sum())
    if diff_cnt > 0:
        diff_stats.append((c, diff_cnt, diff_cnt / len(eq)))

diff_df = pd.DataFrame(diff_stats, columns=["col", "diff_count", "diff_share"]).sort_values(
    ["diff_share", "diff_count"], ascending=False
)

print("Columns with changed values on common keys:", len(diff_df))
print(diff_df.head(30))

# Примеры строк, где есть отличия (для топ-колонки)
if not diff_df.empty:
    top_col = diff_df.iloc[0]["col"]
    s_old = normalize_for_compare(old_common[top_col])
    s_new = normalize_for_compare(new_common[top_col])
    eq = (s_old == s_new) | (s_old.isna() & s_new.isna())
    sample_idx = old_common.index[~eq][:10]
    if len(sample_idx) > 0:
        ex = pd.DataFrame({
            "old": old_common.loc[sample_idx, top_col],
            "new": new_common.loc[sample_idx, top_col],
        })
        print(f"\nSample diffs for column: {top_col}")
        print(ex)

# =========================
# 7) FIRST-DEFAULT ROWS COMPARE (очень полезно для вашей задачи)
# =========================
print("\n=== 7) FIRST DEFAULT ROWS COMPARE ===")

def first_default_rows(df):
    if TARGET_COL not in df.columns:
        return pd.DataFrame(columns=df.columns)
    d = df.sort_values(ID_COLS).copy()
    d["_cum_def"] = d.groupby("vat_num")[TARGET_COL].cumsum()
    out = d[(d[TARGET_COL] == 1) & (d["_cum_def"] == 1)].copy()
    out = out.drop(columns=["_cum_def"])
    return out

fd_old = first_default_rows(df_old)
fd_new = first_default_rows(df_new)

print("First-default rows OLD:", len(fd_old))
print("First-default rows NEW:", len(fd_new))

# сколько из first-default all-zero (по common feature cols)
if len(common_feature_cols) > 0:
    fd_old_zero = fd_old[common_feature_cols].isin([0, "0"]).all(axis=1) if len(fd_old) else pd.Series(dtype=bool)
    fd_new_zero = fd_new[common_feature_cols].isin([0, "0"]).all(axis=1) if len(fd_new) else pd.Series(dtype=bool)
    if len(fd_old):
        print("OLD first-default all-zero:", int(fd_old_zero.sum()), f"({fd_old_zero.mean():.2%})")
    if len(fd_new):
        print("NEW first-default all-zero:", int(fd_new_zero.sum()), f"({fd_new_zero.mean():.2%})")
