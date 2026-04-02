from pandas.tseries.offsets import MonthEnd


def _norm_str(x) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def _pick_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Не найдены колонки {candidates}. Доступные: {list(df.columns)}")


def _pick_predict_col(df: pd.DataFrame, preferred: list[str]) -> str:
    """Пытаемся найти колонку прогноза, даже если имя немного отличается."""
    for c in preferred:
        if c in df.columns:
            return c

    # fallback: любая колонка, содержащая '_predict'
    predict_cols = [c for c in df.columns if "predict" in str(c).lower()]
    if len(predict_cols) == 1:
        return predict_cols[0]
    if predict_cols:
        # если их несколько — берём первую (обычно это нужная)
        return predict_cols[0]

    raise KeyError(f"Не нашёл колонку прогноза. Доступные: {list(df.columns)}")


def _read_group_total_sum(path: Path, pred_date: datetime) -> float:
    """Берём 'Итого' из листа 'Группировка по факторам' (если есть), иначе из первого листа."""
    xls = pd.ExcelFile(path)
    sheet = "Группировка по факторам" if "Группировка по факторам" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet)

    if "Дата" in df.columns:
        df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce")

    factor_col = "factor" if "factor" in df.columns else ("Фактор" if "Фактор" in df.columns else None)
    if factor_col is None:
        raise KeyError(f"Не нашёл колонку factor/Фактор в {path}")

    df_f = df.loc[df[factor_col].astype(str).str.lower().str.strip() == "итого"]
    if "Дата" in df_f.columns:
        df_f = df_f.loc[df_f["Дата"] == pd.Timestamp(pred_date)]

    if df_f.empty:
        raise ValueError(f"Не найдена строка Итого на дату {pred_date.date()} в {path}")

    sum_col = _pick_first_existing_col(
        df_f,
        [
            "Сумма МСФО_predict_БМ",
            "Сумма МСФО_predict",
            "Сумма МСФО",
        ],
    )

    return float(df_f.iloc[0][sum_col])


def _read_base_company_pred(path: Path, company_name: str, pred_date: datetime) -> float:
    xls = pd.ExcelFile(path)
    sheet = "data" if "data" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet)

    if "Дата" in df.columns:
        df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce")

    company_col = "Компания" if "Компания" in df.columns else None
    if company_col is None:
        raise KeyError(f"Не нашёл колонку Компания в {path}")

    df_c = df.loc[df[company_col].map(_norm_str) == _norm_str(company_name)]
    if "Дата" in df_c.columns:
        df_c = df_c.loc[df_c["Дата"] == pd.Timestamp(pred_date)]

    if df_c.empty:
        raise ValueError(f"Не найдена компания {company_name!r} на дату {pred_date.date()} в {path}")

    pred_col = _pick_predict_col(df_c, ["НЗП, млрд. руб._predict"])
    return float(df_c.iloc[0][pred_col])


BASE_COMPANIES_FOR_YML = ["ГРКБ", "МР", "Прочие"]

records = []
errors = []

for month_dir in month_dirs:
    month_dt = parse_month_from_folder(month_dir.name)
    pred_date = month_dt + MonthEnd(0)
    archive_dir = ARCHIVE_ROOT / month_dir.name

    zf_path = archive_dir / "predict_NZP_ZF_БМ.xlsx"
    kgmk_path = archive_dir / "predict_NZP_KGMK_БМ.xlsx"
    base_path = archive_dir / "NZP_BPC_BASE_predict.xlsx"

    try:
        if not zf_path.exists() or not kgmk_path.exists() or not base_path.exists():
            missing = [p.name for p in [zf_path, kgmk_path, base_path] if not p.exists()]
            raise FileNotFoundError(f"В архиве {archive_dir} нет файлов: {missing}")

        y_basemodel = _read_base_company_pred(base_path, "Группа компаний", pred_date)

        zf_total = _read_group_total_sum(zf_path, pred_date)
        kgmk_total = _read_group_total_sum(kgmk_path, pred_date)

        base_vals = {c: _read_base_company_pred(base_path, c, pred_date) for c in BASE_COMPANIES_FOR_YML}
        base_other_sum = sum(base_vals.values())

        y_ml = zf_total + kgmk_total + base_other_sum

        records.append(
            {
                "Дата": pd.Timestamp(pred_date),
                "y_BaseModel": y_basemodel,
                "y_ML": y_ml
            }
        )

    except Exception as e:
        errors.append((month_dir.name, str(e)))


df_summary = pd.DataFrame(records).sort_values("Дата").reset_index(drop=True)
print(f"Собрано месяцев: {len(df_summary)}")
display(df_summary)

if errors:
    print("\nОшибки при сборке сводной таблицы:")
    for m, err in errors:
        print("-", m)
        print(err)
