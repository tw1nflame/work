import os
import re
import scipy
import shutil
import logging
import numpy as np
import pandas as pd
import streamlit as st
from functools import reduce

from pandas.tseries.offsets import MonthEnd, MonthBegin

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.linear_model import LinearRegression

from utils.datahandler import *
from utils.predictor import *
from utils.common import *
from utils.pipelines import *
from utils.excel_formatter import *
from utils.factoring import *

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

logger = setup_custom_logging("log.txt")


def log_pretrained_models_directory_state(logger):
    try:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        pretrained_models_dir = os.path.join(app_dir, "pretrained_models")

        logger.info(f"Checking pretrained models directory: {pretrained_models_dir}")

        if not os.path.exists(pretrained_models_dir):
            logger.warning(f"Pretrained models directory does not exist: {pretrained_models_dir}")
            return

        if not os.path.isdir(pretrained_models_dir):
            logger.error(f"Pretrained models path exists, but is not a directory: {pretrained_models_dir}")
            return

        logger.info(f"Pretrained models directory exists: {pretrained_models_dir}")

        subdirs = []
        try:
            with os.scandir(pretrained_models_dir) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            subdirs.append(entry.name)
                    except Exception:
                        logger.exception(f"Failed to inspect entry in pretrained models directory: {entry.path}")
        except Exception:
            logger.exception(f"Failed to scan pretrained models directory: {pretrained_models_dir}")
            return

        subdirs.sort()
        logger.info(f"Pretrained models subdirectories ({len(subdirs)}): {subdirs}")

        for subdir in subdirs:
            subdir_path = os.path.join(pretrained_models_dir, subdir)
            child_dirs = []
            child_files = []
            try:
                with os.scandir(subdir_path) as entries:
                    for entry in entries:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                child_dirs.append(entry.name)
                            elif entry.is_file(follow_symlinks=False):
                                child_files.append(entry.name)
                            else:
                                logger.info(f"Skipping non-file entry in pretrained model subdirectory: {entry.path}")
                        except Exception:
                            logger.exception(f"Failed to inspect entry in pretrained model subdirectory: {entry.path}")
            except Exception:
                logger.exception(f"Failed to scan pretrained model subdirectory: {subdir_path}")
                continue

            child_dirs.sort()
            child_files.sort()
            logger.info(
                f"Pretrained model subdirectory '{subdir}' contains "
                f"subdirectories ({len(child_dirs)}): {child_dirs}; "
                f"files ({len(child_files)}): {child_files}"
            )
    except Exception:
        logger.exception("Unexpected error while checking pretrained models directory")


log_pretrained_models_directory_state(logger)

with st.sidebar:
    with st.expander("tech user:"):
        st.text_input("username:", key="tech_user")
        st.text_input("password:", key="tech_password", type="password")
    if (st.session_state.get("tech_user", False) == "admin") and (st.session_state.get("tech_password", False) == "admin"):
        if os.path.exists('log.txt'):
            with open('log.txt', 'rb') as f:
                st.sidebar.download_button(
                    label='Download Logs',
                    data=f,
                    file_name='log.txt',
                    mime='text/plain'
                )

left, mid, right = st.columns(3)

with left:
    BPC_file = st.file_uploader(
        "Загрузите файл 'НЗП общий свод плоский файл.xlsx'", 
        key="BPC_file",
        type=["xlsm", "xlsx"]
    )
    
    st.divider()
    
    KP_ZF_file = st.file_uploader(
        "Загрузите файл 'КП НЗП ГМК плоский файл.xlsx'",
        key="KP_ZF_file",
        type=["xls", "xlsx"]
    )

    st.divider()

    KP_KGMK_file = st.file_uploader(
        "Загрузите файл 'КП НЗП КГМК плоский файл.xlsx'",
        key="KP_KGMK_file",
        type=["xlsx", "xls"]
    )

    st.divider()
    
    stein_file = st.file_uploader(
        "Загрузите файл 'Штейн никелевый факт НЗП.xlsx'",
        key="stein_file",
        type=["xlsx", "xls"]
    )
    
with mid:
    finestein_file = st.file_uploader(
        "Загрузите файл 'Файнштейн ЗФ КГМК плоский файл.xlsx'",
        key="finestein_file",
        type=["xlsx", "xls"]
    )
    
    st.divider()
    
    NNH_file = st.file_uploader(
        "Загрузите файл 'ННХ плоский файл.xlsx'", 
        key="NNH_file",
        type=["xlsx", "xls"]
    )
    
    st.divider()
    
    factor_maping = st.file_uploader(
        "Загрузите файл 'Позиции НЗП с признаком ГП для группы полупродукты нзп гп.xlsx'", 
        key="factor_maping",
        type=["xlsx", "xls"]
    )

    st.divider()

    results_file = st.file_uploader(
        "Загрузите файл 'Результаты ML_basemodel_coeff_.xlsx'", 
        key="results_file",
        type=["xlsx", "xls"]
    )
    
with right:
    BM_file = st.file_uploader(
        "Загрузите файл 'БМ плоский файл.xlsx'", 
        key="BM_file",
        type=["xlsx", "xls"]
    )
    
    st.divider()

    correction_file = st.file_uploader(
        "Загрузите файл 'Корректировки.xlsx'", 
        key="correction_file",
        type=["xlsx", "xls"]
    )
    
    st.divider()

    KGMK_concentrates_file = st.file_uploader(
        "Загрузите файл 'Концентраты КГМК плоский файл.xlsx'", 
        key="KGMK_concentrates_file",
        type=["xlsx", "xls"]
    )


input_date = st.date_input(
    "Выберите месяц и год предикта (число можно игнорировать, возьмётся 1-е число)",
    value=datetime(year=datetime.now().year, month=datetime.now().month, day=1)
)

# Приведём выбранную дату к 1 числу месяца (учитывая год/месяц из input_date)
CHOSEN_MONTH = datetime(input_date.year, input_date.month, 1)
st.write(f"Предиктивный месяц: {CHOSEN_MONTH.strftime('%B %Y')}")

if st.button("Запустить расчёт"):
    logger.info("Calculation started")
    # Проверяем, что все файлы загружены
    if not all([BPC_file, KP_ZF_file, KP_KGMK_file, finestein_file, NNH_file, factor_maping, BM_file, correction_file]):
        logger.error("Not all files was uploaded")
        st.error("Необходимо загрузить все файлы")
        st.stop()

    time_start = datetime.now()
    with st.spinner("Идет прогнозирование..."):
        # ЗФ пайплайн
        ZF_pipeline(
            BPC_file=BPC_file,
            KP_file=KP_ZF_file,
            factoring_maping_file=factor_maping,
            finestein_file=finestein_file,
            NNH_file=NNH_file,
            BM_file=BM_file,
            correction_file=correction_file,
            MONTH_TO_PREDICT=CHOSEN_MONTH
        )
        
        # КГМК пайплайн
        KGMK_pipeline(
            BPC_file=BPC_file,
            KP_file=KP_KGMK_file,
            factoring_maping_file=factor_maping,
            finestein_file=finestein_file,
            stein_file=stein_file,
            KGMK_concentrates=KGMK_concentrates_file,
            BM_file=BM_file,
            correction_file=correction_file,
            MONTH_TO_PREDICT=CHOSEN_MONTH
        )
    
        BPC_pipeline(
            BPC_file=BPC_file,
            MONTH_TO_PREDICT=CHOSEN_MONTH
        )
    
        st.session_state['linreg_result'] = get_linreg_weights(
            results_file=results_file,
            MONTH_TO_PREDICT=CHOSEN_MONTH
        )

    runtime = datetime.now() - time_start
    st.session_state['runtime'] = runtime
    logger.info(f"Pipeline completed in {runtime}")
    st.session_state['result_ready'] = True
    logger.info("Results ready")

if st.session_state.get('result_ready', False):
    st.success(f"Прогнозирование выполнено успешно. Время выполнения: {st.session_state['runtime']}")
    st.info(st.session_state['linreg_result'])
    
    with open('results/predict_NZP_ZF_БМ.xlsx', "rb") as result_file:
        res = result_file.read()
        st.download_button(
            label='Скачать прогнозный файл НЗП ЗФ',
            data=res,
            file_name='results/predict_NZP_ZF_БМ.xlsx'
        )

    with open('results/predict_NZP_KGMK_БМ.xlsx', "rb") as result_file:
        res = result_file.read()
        st.download_button(
            label='Скачать прогнозный файл НЗП КГМК',
            data=res,
            file_name='results/predict_NZP_KGMK_БМ.xlsx'
        )

    with open('results/NZP_BPC_BASE_predict.xlsx', "rb") as result_file:
        res = result_file.read()
        st.download_button(
            label='Скачать прогнозный файл Базовый по компаниям',
            data=res,
            file_name='results/NZP_BPC_BASE_predict.xlsx'
        )
