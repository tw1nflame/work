import os
import re
import json
import scipy
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

from pandas.tseries.offsets import MonthEnd

from datetime import datetime, timedelta
from functools import reduce
from dateutil.relativedelta import relativedelta
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

import logging

def _normalize_item_id_value(value):
    """Normalize item_id values to str or None.

    AutoGluon TimeSeries requires id_column to have integer or string dtype.
    In practice, Excel/pandas merges may introduce NaN or float values.
    """
    if pd.isna(value):
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized or normalized.lower() == "nan":
            return None
        return normalized
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        # Convert 123.0 -> "123" to keep stable ids
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    normalized = str(value).strip()
    if not normalized or normalized.lower() == "nan":
        return None
    return normalized

def normalize_to_list(value):
    """Normalize a scalar or iterable to a Python list.

    Used to accept both a single target column name and a list of target names.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]

def predict_naive(df, MONTHES_TO_PREDICT, FACTORS, TARGETS, REQUIRED_COLUMNS):
    RESULT_DF_NAIVE = []
    logging.info(f'Прогнозирование {FACTORS} по наиву')
    
    for FACTOR in FACTORS:
        df_prep = df.loc[(df['factor'] == FACTOR)]
        if df_prep.empty:
            logging.warning(f'⚠️ Sub df for factor {FACTOR} is empty. Skipping prediction for {FACTOR}')
            continue
        
        for MONTH in MONTHES_TO_PREDICT:
            MONTH_TO_PREDICT_first_day = MONTH
            MONTH_TO_PREDICT_last_day = MONTH_TO_PREDICT_first_day + MonthEnd(0)
            MONTH_TO_PREDICT_previous = MONTH_TO_PREDICT_first_day - relativedelta(months=1) + MonthEnd(0)
            
            df_naive = df_prep.loc[df_prep['Дата'] == MONTH_TO_PREDICT_previous]
            
            TARGETS_rename = {target: f'{target}_predict' for target in TARGETS}
            df_naive = df_naive.rename(columns=TARGETS_rename)
            df_naive['Дата'] = MONTH_TO_PREDICT_last_day
            MONTH_PRED = df_naive.loc[:, REQUIRED_COLUMNS]
            
            for TARGET in TARGETS:
                columns_to_use = [*REQUIRED_COLUMNS, TARGETS_rename[TARGET]]
                MONTH_PRED = pd.merge(MONTH_PRED, df_naive[columns_to_use], on=REQUIRED_COLUMNS)
            RESULT_DF_NAIVE.append(MONTH_PRED)
            
    if RESULT_DF_NAIVE:
        return pd.concat(RESULT_DF_NAIVE)
    else:
        return pd.DataFrame()

def prepare_time_series_data(df_train, prediction_date, date_column):
    """
    Prepare time series data for training by converting to TimeSeriesDataFrame format
    and filling missing values.
    
    Args:
        df_train: DataFrame containing training data for a specific item_id
        prediction_date: The date for which prediction is being made
        
    Returns:
        TimeSeriesDataFrame ready for model training
    """
    if df_train is None or df_train.empty:
        return TimeSeriesDataFrame.from_data_frame(
            pd.DataFrame({"item_id": pd.Series(dtype="string"), date_column: pd.Series(dtype="datetime64[ns]")}),
            id_column="item_id",
            timestamp_column=date_column,
        )

    df_train = df_train.copy()
    # Ensure proper dtypes for AutoGluon validation
    df_train["item_id"] = df_train["item_id"].map(_normalize_item_id_value)
    df_train = df_train.dropna(subset=["item_id"])
    df_train["item_id"] = df_train["item_id"].astype("string")
    df_train[date_column] = pd.to_datetime(df_train[date_column], errors="coerce")
    df_train = df_train.dropna(subset=[date_column])

    if df_train.empty:
        return TimeSeriesDataFrame.from_data_frame(
            pd.DataFrame({"item_id": pd.Series(dtype="string"), date_column: pd.Series(dtype="datetime64[ns]")}),
            id_column="item_id",
            timestamp_column=date_column,
        )

    # Convert to TimeSeriesDataFrame format
    df_ts = TimeSeriesDataFrame.from_data_frame(
        df_train,
        id_column="item_id",
        timestamp_column=date_column
    )
    
    df_ts = df_ts.reset_index()
    
    # Fill gaps in time series with zeros
    last_month = prediction_date - MonthEnd(1)
    full_index = pd.MultiIndex.from_product([
        df_ts['item_id'].unique(),
        pd.date_range(df_ts['timestamp'].min(), last_month, freq='M')
    ], names=['item_id', 'timestamp'])
    
    df_ts = df_ts.set_index(['item_id', 'timestamp']).reindex(full_index, fill_value=0).convert_frequency(freq="M")
    df_ts = df_ts.fillna(0)
    
    return df_ts

def train_and_predict_model(df_ts, target_column, model_path, metric, factor, ARIMA_seasonal):
    """
    Train a time series model and make predictions.
    
    Args:
        df_ts: TimeSeriesDataFrame containing the training data
        target_column: The column to predict
        model_path: Path where the model will be saved
        metric: Evaluation metric to use
        ARIMA_seasonal: Hyperparameter  for AutoARIMA model in ensemble. True by default
        
    Returns:
        tuple: (predictions DataFrame, model information)
    """
    # Initialize and train the predictor
    df_train = df_ts.loc[:, [target_column]]

    model_dir = Path(model_path)
    if model_dir.exists() and model_dir.is_dir():
        shutil.rmtree(model_dir)

    predictor = TimeSeriesPredictor(
        prediction_length=1,
        path=model_path,
        target=target_column,
        eval_metric=metric,
        freq="M"
    )
    
    models = {
        'Naive': {},   
        'CrostonSBA': {},
        'AutoETS': {}, 
        'DynamicOptimizedTheta': {},
        'AutoARIMA': {},
        'Chronos': 
              [
                  {
                      'model_path': 'pretrained_models/chronos-bolt-base',
                      'ag_args': {'name_suffix': 'ZeroShot'}
                  },
                  {
                      'model_path': 'pretrained_models/chronos-bolt-small',
                      'ag_args': {'name_suffix': 'ZeroShot'}
                  }
              ],
          'Chronos2': 
            [
                {'model_path': 'pretrained_models/chronos-2',
                'ag_args': {'name_suffix': 'ZeroShot'}
                 },
                {
                    'model_path': 'pretrained_models/chronos-2-small',
                  'ag_args': {'name_suffix': 'ZeroShot'}
                }
            ],
        'PatchTSTModel': {},
        'TemporalFusionTransformerModel': {}
    }

    # models = {
#           'NaiveModel': {},
#           'SeasonalNaiveModel': {},
#           'AverageModel': {},
#           'SeasonalAverageModel': {},
#           'ZeroModel': {},
#           'AutoARIMAModel': {},
#           'AutoETSModel': {},
#           'ThetaModel': {},
#           'CrostonModel': {},
#           'DynamicOptimizedTheta': {},
#           'NPTSModel': {},
#           'DeepARModel': {},
#           'PatchTSTModel': {},
#           'TemporalFusionTransformerModel': {},
#           'TiDEModel': {},
#           'DirectTabularModel': {},
#           'RecursiveTabularModel': {},
#           'Chronos': 
#               [
#                   {
#                       'model_path': 'pretrained_models/chronos-bolt-base',
#                       'ag_args': {'name_suffix': 'ZeroShot'}
#                   },
#                   {
#                       'model_path': 'pretrained_models/chronos-bolt-small',
#                       'ag_args': {'name_suffix': 'ZeroShot'}
#                   },
#                   {
#                       'model_path': 'pretrained_models/chronos-bolt-small',
#                       'fine_tune': True,
#                       'ag_args': {'name_suffix': 'FineTuned'}
#                   }
#               ],
#           'Chronos2': 
#             [
#                 {'model_path': 'pretrained_models/chronos-2',
#                 'ag_args': {'name_suffix': 'ZeroShot'}
#                  },
#                 {
#                     'model_path': 'pretrained_models/chronos-2-small',
#                   'ag_args': {'name_suffix': 'ZeroShot'}
#                 },
#                 {
#                     'model_path': 'pretrained_models/chronos-2-small',
#                     'fine_tune': True,
#                     'ag_args': {'name_suffix': 'FineTuned'}
#                 }
#             ]
#           }
    
    if not ARIMA_seasonal:
        models['AutoARIMA'] = {'seasonal': False}

    if (factor == 'Годовая сезонность') & (target_column == 'Количество'):
        models = {'SeasonalNaive': {'seasonal_period': 12}}
    
    predictor.fit(
        df_train,
        presets="high_quality",
        hyperparameters=models,
    )
    
    # Extract model information
    if len(models) == 1:
        model_info = list(models.keys())[0]
    else:
        model_info = predictor.info()['model_info'].copy()
        for model_data in model_info.values():
            del model_data['quantile_levels']
            if model_data['name'] != 'WeightedEnsemble':
                del model_data['info_per_val_window']
    
    # Make predictions
    try:
        predictions = predictor.predict(df_train)
        
        models_dir = Path('models')
        
        if models_dir.exists() and models_dir.is_dir():
            shutil.rmtree(models_dir)
            
    except Exception as E:
        logging.error(f'Не удалось сделать прогноз из-за ошибки: {E}')
        predictions = pd.DataFrame()
        model_info = None
    
    return predictions, model_info

def format_predictions(predictions, target_column, prediction_date, factor, date_column):
    """
    Format the predictions into a standardized DataFrame.
    
    Args:
        predictions: Raw predictions from the model
        target_column: The column that was predicted
        prediction_date: The date for which prediction was made
        factor: The factor used for prediction
        
    Returns:
        DataFrame with formatted predictions
    """
    result = predictions.reset_index().loc[:, ["item_id", "timestamp", "mean"]]
    result = result.rename(columns={"timestamp": date_column, "mean": f"{target_column}_predict"})
    result[date_column] = prediction_date
    
    return result

def generate_timeseries_predictions(
    df, 
    months_to_predict, 
    metric, 
    factors, 
    targets,
    date_column,
    company,
    ARIMA_seasonal=True
):
    """
    Make time series predictions for multiple factors, months, and targets.
    
    Args:
        df: DataFrame containing the data
        months_to_predict: List of months for which to make predictions
        metric: Evaluation metric to use
        factors: List of factors to predict for
        targets: List of target columns to predict
        company: Company identifier for model path
        
    Returns:
        tuple: (DataFrame with all predictions, dictionary with model information)
    """
    result_dfs = []
    models_info_dfs = []

    if not isinstance(targets, list):
        logging.info(f"Преобразовали targets в список, был {type(targets)}")
        targets = normalize_to_list(targets)
    
    logging.info(f'Company: {company}')
    for factor in factors:
        logging.info(f'Factor: {factor}')
        
        # Initialize factor info in the model info dictionary
        df_factor = df.loc[df['factor'] == factor]
        
        if df_factor.empty:
            logging.warning(f'⚠️ Sub dataframe for company - {company}; factor - {factor} is empty. Skipping prediction for {company} - {factor}')
            continue
        
        for month in months_to_predict:
            logging.info(f'\tMonth: {month}')
            
            # Format date for model info dictionary
            date_str = month.strftime('%Y-%m-%d')
            
            # Define prediction period
            prediction_date = month + MonthEnd(0)
            
            # Filter training data
            df_train = df_factor.loc[df_factor[date_column] < month]
            df_train = df_train.loc[:, ['item_id', date_column] + targets].reset_index(drop=True)

            if df_train.empty:
                logging.warning(
                    f"⚠️ Нет обучающих данных для company={company}; factor={factor}; month={month}. "
                    "Пропускаю прогноз для этого месяца."
                )
                continue

            # Защита от битых item_id (NaN/float/пустые): AutoGluon требует int/str
            df_train['item_id'] = df_train['item_id'].map(_normalize_item_id_value)
            df_train = df_train.dropna(subset=['item_id'])
            if df_train.empty:
                logging.warning(
                    f"⚠️ После очистки item_id не осталось строк для company={company}; factor={factor}; month={month}. "
                    "Пропускаю прогноз для этого месяца."
                )
                continue

            df_ts = prepare_time_series_data(df_train, month, date_column)

            if df_ts is None or len(df_ts) == 0:
                logging.warning(
                    f"⚠️ Пустой TimeSeriesDataFrame для company={company}; factor={factor}; month={month}. "
                    "Пропускаю прогноз для этого месяца."
                )
                continue
            
            TARGET_PREDICTS = []
            for target_column in targets:
                logging.info(f'\t\tTarget: {target_column}')
                model_path = f"models/{company}_{factor}_{target_column}_{prediction_date.strftime('%B%Y')}_{metric}"
                
                # Train model and make predictions
                predictions, model_info = train_and_predict_model(
                    df_ts, 
                    target_column, 
                    model_path, 
                    metric,
                    factor,
                    ARIMA_seasonal
                )
                
                if model_info:
                    if isinstance(model_info, dict):
                        model_weights = model_info.get("WeightedEnsemble", {}).get("model_weights", {})
                        ensemble_info = {k: round(v, 4) for k, v in model_weights.items()}
                    elif isinstance(model_info, str):
                        ensemble_info = {model_info: 1.0}
                    else:
                        ensemble_info = 'Unknown'
                        
                    record = {
                        'Компания': company,
                        'Фактор': factor,
                        date_column: date_str,
                        'Таргет': target_column,
                        'Ансамбль': [ensemble_info]
                    }
                    models_info_dfs.append(pd.DataFrame(record))

                if not predictions.empty:
                    # Format and store predictions
                    result = format_predictions(predictions, target_column, prediction_date, factor, date_column)
                    TARGET_PREDICTS.append(result)

            if not TARGET_PREDICTS:
                logging.warning(
                    f"⚠️ Не удалось получить предикты ни по одному таргету для company={company}; factor={factor}; month={month}. "
                    "Пропускаю запись результата за месяц."
                )
                continue

            MONTH_PREDICT = reduce(
                lambda left, right: pd.merge(left, right, on=['item_id', date_column], how='left'),
                TARGET_PREDICTS,
            )
            MONTH_PREDICT['factor'] = factor
            result_dfs.append(MONTH_PREDICT)
            
    # Combine all predictions
    if result_dfs:
        models_info = pd.concat(models_info_dfs) if models_info_dfs else pd.DataFrame()
        return pd.concat(result_dfs), models_info
    else:
        return pd.DataFrame(), pd.DataFrame()

def get_finestein_predict(finestein_file, COMPANY_ru, MONTHES_TO_PREDICT):
    FINESTEIN_predict = pd.read_excel(finestein_file)
    FINESTEIN_predict = FINESTEIN_predict.loc[FINESTEIN_predict['item_id'] == f'Файнштейн {COMPANY_ru}']
    FINESTEIN_predict['item_id'] = 'Файнштейн ГП'
    FINESTEIN_predict['factor'] = f'Файнштейн {COMPANY_ru}'
    FINESTEIN_predict['Дата'] += MonthEnd(0)
    # Выбираем только месяцы, заданные для предикта
    FINESTEIN_predict = FINESTEIN_predict.loc[FINESTEIN_predict['Дата'].isin(list(map(lambda x: x + MonthEnd(0), MONTHES_TO_PREDICT)))]
    FINESTEIN_predict['Дата'] = FINESTEIN_predict['Дата'].astype('datetime64[us]')
    
    return FINESTEIN_predict
