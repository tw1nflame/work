import pandas as pd
from typing import Dict, List, Union

def add_lagged_covariates(
    df: pd.DataFrame, 
    date_column: str,
    lags: Dict[str, Union[int, List[int]]]
) -> pd.DataFrame:
    """
    Добавляет лагированные признаки к датафрейму, удаляя строки с образовавшимися пропусками
    (чтобы можно было безопасно передать в прогнозные модели).
    
    Параметры:
    ----------
    df : pd.DataFrame
        Исходный датафрейм, содержащий дату и признаки.
    date_column : str
        Название колонки с датой. Обязательна для корректной сортировки перед сдвигом.
    lags : Dict[str, Union[int, List[int]]]
        Словарь, где ключ - название признака, а значение - лаг или список лагов (в периодах).
        Пример: {'row1': [1, 2, 3], 'row2': 1}
        
    Возвращает:
    ----------
    pd.DataFrame
        Новый датафрейм с добавленными колонками вида `row1_lag1` 
        и обрезанными пустыми строками в начале.
    """
    df_res = df.copy()
    
    # Обязательно сортируем по дате, чтобы лаги брались из прошлого
    df_res = df_res.sort_values(date_column)
    
    for col, lag_vals in lags.items():
        if col not in df_res.columns:
            print(f"⚠️ Предупреждение: колонка {col} не найдена в датафрейме. Лаги не добавлены.")
            continue
            
        if isinstance(lag_vals, int):
            lag_vals = [lag_vals]
            
        for lag in lag_vals:
            col_name = f"{col}_lag{lag}"
            df_res[col_name] = df_res[col].shift(lag)
            
    # Удаляем строки, где образовались NaN из-за сдвига
    df_res = df_res.dropna()
    df_res = df_res.reset_index(drop=True)
    
    return df_res

from pandas.tseries.offsets import MonthEnd

def _ensure_future_row(df: pd.DataFrame, date_column: str, target_column: Union[str, List[str]] = None) -> pd.DataFrame:
    """
    Вспомогательная функция. Проверяет, есть ли в датафрейме строка, 
    соответствующая следующему месяцу после максимальной даты.
    Если нет (или если последняя строка не является пустой прогнозной), добавляет её. 
    Помогает подготовить known_covariates для прогноза.
    """
    last_row = df.iloc[-1]
    
    # Проверяем, является ли уже последняя строка "пустой" прогнозной 
    is_future_row = False
    if target_column is not None:
        t_cols = [target_column] if isinstance(target_column, str) else target_column
        valid_t_cols = [c for c in t_cols if c in df.columns]
        if valid_t_cols and pd.isna(last_row[valid_t_cols]).all():
            is_future_row = True
    else:
        # Fallback: все колонки кроме даты и сгенерированных фичей
        original_cols = [c for c in df.columns if c != date_column and c != 'month' and not str(c).startswith('is_')]
        if original_cols and pd.isna(last_row[original_cols]).all():
            is_future_row = True
            
    if is_future_row:
        # Последняя строка уже была добавлена ранее (в ней нет исторических данных),
        # поэтому новую создавать не нужно
        return df

    max_date = df[date_column].max()
    
    from dateutil.relativedelta import relativedelta
    next_month = max_date + relativedelta(months=1) + MonthEnd(0)
    
    if not (df[date_column] == next_month).any():
        new_row = {date_column: next_month}
        # Заполняем остальные колонки NaN
        for col in df.columns:
            if col != date_column:
                new_row[col] = pd.NA
                
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df

def add_month_feature(df: pd.DataFrame, date_column: str, feature_name: str = 'month', target_column: Union[str, List[str]] = None) -> pd.DataFrame:
    """
    Добавляет номер месяца в качестве признака (извлекает его из колонки с датой).
    Также гарантирует наличие строки для следующего прогнозного месяца.
    
    Параметры:
    ----------
    df : pd.DataFrame
        Исходный датафрейм, содержащий дату.
    date_column : str
        Название колонки с датой.
    feature_name : str
        Название новой колонки. По умолчанию 'month'.
    target_column : Union[str, List[str]], optional
        Колонки, по которым проверяется, пустая ли уже строка.
        
    Возвращает:
    ----------
    pd.DataFrame
        Новый датафрейм с добавленной колонкой месяца.
    """
    df_res = df.copy()
    
    if date_column not in df_res.columns:
        raise ValueError(f"Колонка {date_column} не найдена в датафрейме.")
        
    if not pd.api.types.is_datetime64_any_dtype(df_res[date_column]):
        df_res[date_column] = pd.to_datetime(df_res[date_column])
        
    df_res = _ensure_future_row(df_res, date_column, target_column=target_column)
        
    df_res[feature_name] = df_res[date_column].dt.month
    
    return df_res

import calendar

def is_special_month(df: pd.DataFrame, date_column: str, months: List[int], target_column: Union[str, List[str]] = None) -> pd.DataFrame:
    """
    Добавляет бинарные колонки-индикаторы для указанных месяцев (напр., is_january, is_february).
    Также гарантирует наличие строки для следующего прогнозного месяца.
    
    Параметры:
    ----------
    df : pd.DataFrame
        Исходный датафрейм, содержащий дату.
    date_column : str
        Название колонки с датой.
    months : List[int]
        Список номеров месяцев (1-12), для которых нужно добавить признаки.
        Например: [1, 12] добавит колонки is_january и is_december.
    target_column : Union[str, List[str]], optional
        Колонки, по которым проверяется, пустая ли уже строка. 
        
    Возвращает:
    ----------
    pd.DataFrame
        Новый датафрейм с добавленными бинарными признаками.
    """
    df_res = df.copy()
    
    if date_column not in df_res.columns:
        raise ValueError(f"Колонка {date_column} не найдена в датафрейме.")
        
    if not pd.api.types.is_datetime64_any_dtype(df_res[date_column]):
        df_res[date_column] = pd.to_datetime(df_res[date_column])
        
    df_res = _ensure_future_row(df_res, date_column, target_column=target_column)
        
    for month_num in months:
        if 1 <= month_num <= 12:
            month_name = calendar.month_name[month_num].lower()
            col_name = f"is_{month_name}"
            df_res[col_name] = (df_res[date_column].dt.month == month_num).astype(int)
        else:
            print(f"⚠️ Предупреждение: Некорректный номер месяца {month_num}. Пропускаем.")
            
    return df_res


