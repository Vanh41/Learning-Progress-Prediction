import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
EXTERNAL_DATA_DIR = DATA_DIR / 'external'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

MODELS_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'output'
EVALUATION_DIR = OUTPUT_DIR / 'evaluation'

for directory in [RAW_DATA_DIR, EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUT_DIR, EVALUATION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def get_model_output_dir(model_name):
    model_dir = EVALUATION_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir

ADMISSION_FILE = RAW_DATA_DIR / 'admission.csv'
ACADEMIC_RECORDS_FILE = RAW_DATA_DIR / 'academic_records.csv'
TEST_FILE = RAW_DATA_DIR / 'test.csv'

TRAIN_START_YEAR = 2020
TRAIN_END_SEMESTER = 'HK1 2023-2024'  
VALID_SEMESTER = 'HK2 2023-2024'       
TEST_SEMESTER = 'HK1 2024-2025'       

RANDOM_STATE = 42
N_SPLITS = 5  

DEFAULT_PARAMS = {
    'xgboost': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'random_state': RANDOM_STATE
    },
    'lightgbm': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'random_state': RANDOM_STATE
    },
    'catboost': {
        'depth': 6,
        'learning_rate': 0.1,
        'iterations': 100,
        'random_state': RANDOM_STATE,
        'verbose': False
    }
}

OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT = 3600 

FEATURE_GROUPS = {
    'admission': ['NAM_TUYENSINH', 'PTXT', 'TOHOP_XT', 'DIEM_TRUNGTUYEN', 'DIEM_CHUAN'],
    'academic': ['HOC_KY', 'CPA', 'GPA', 'TC_DANGKY', 'TC_HOANTHANH'],
    'engineered': []  
}

METRICS = ['R2', 'RMSE', 'MSE', 'MAPE']