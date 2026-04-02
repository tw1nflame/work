def _read_ensemble_sheet(xlsx_path: Path, preferred_sheet: str) -> pd.DataFrame:
    xls = pd.ExcelFile(xlsx_path)
    if preferred_sheet in xls.sheet_names:
        return pd.read_excel(xlsx_path, sheet_name=preferred_sheet)

    # fallback: иногда лист может быть иначе назван (на всякий случай)
    for s in xls.sheet_names:
        if _norm_str(s) == _norm_str(preferred_sheet):
            return pd.read_excel(xlsx_path, sheet_name=s)

    raise KeyError(f"Лист {preferred_sheet!r} не найден в {xlsx_path}. Есть: {xls.sheet_names}")


final_out = REPO_ROOT / "results" / "batch_summary_with_ensembles.xlsx"

ensemble_records = []
ensemble_errors = []

for month_dir in month_dirs:
    month_dt = parse_month_from_folder(month_dir.name)
    pred_date = month_dt + MonthEnd(0)
    archive_dir = ARCHIVE_ROOT / month_dir.name

    zf_path = archive_dir / "predict_NZP_ZF_БМ.xlsx"
    kgmk_path = archive_dir / "predict_NZP_KGMK_БМ.xlsx"
    base_path = archive_dir / "NZP_BPC_BASE_predict.xlsx"

    # ZF ensembles
    try:
        if zf_path.exists():
            df = _read_ensemble_sheet(zf_path, "Ансамбли")
            df.insert(0, "Месяц_папка", month_dir.name)
            df.insert(1, "Дата", pd.Timestamp(pred_date))
            df.insert(2, "Pipeline", "ZF")
            ensemble_records.append(df)
    except Exception as e:
        ensemble_errors.append((month_dir.name, "ZF", str(e)))

    # KGMK ensembles
    try:
        if kgmk_path.exists():
            df = _read_ensemble_sheet(kgmk_path, "Ансамбли")
            df.insert(0, "Месяц_папка", month_dir.name)
            df.insert(1, "Дата", pd.Timestamp(pred_date))
            df.insert(2, "Pipeline", "KGMK")
            ensemble_records.append(df)
    except Exception as e:
        ensemble_errors.append((month_dir.name, "KGMK", str(e)))

    # BASE ensembles
    try:
        if base_path.exists():
            df = _read_ensemble_sheet(base_path, "ensemble_info")
            df.insert(0, "Месяц_папка", month_dir.name)
            df.insert(1, "Дата", pd.Timestamp(pred_date))
            df.insert(2, "Pipeline", "BASE")
            ensemble_records.append(df)
    except Exception as e:
        ensemble_errors.append((month_dir.name, "BASE", str(e)))


if ensemble_records:
    df_ensembles_all = pd.concat(ensemble_records, ignore_index=True)
else:
    df_ensembles_all = pd.DataFrame()

# Пишем итоговый файл
with pd.ExcelWriter(final_out, engine="openpyxl") as writer:
    # summary
    if "df_summary" in globals() and isinstance(df_summary, pd.DataFrame):
        df_summary.to_excel(writer, sheet_name="summary", index=False)
    else:
        pd.DataFrame().to_excel(writer, sheet_name="summary", index=False)

    # ensembles
    df_ensembles_all.to_excel(writer, sheet_name="ensembles", index=False)

    # errors (если есть)
    if ensemble_errors:
        pd.DataFrame(ensemble_errors, columns=["Месяц_папка", "Pipeline", "Ошибка"]).to_excel(
            writer, sheet_name="ensemble_errors", index=False
        )

print("Итоговый файл сохранён:", final_out)
print("Ансамблей строк:", len(df_ensembles_all))
if ensemble_errors:
    print("Ошибок чтения ансамблей:", len(ensemble_errors))
