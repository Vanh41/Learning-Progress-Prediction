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
        'learning_rate': 0.05,
        'n_estimators': 200,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_lambda': 0.1,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    'lightgbm': {
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.5,
        'reg_lambda': 1.0,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1
    },
    'catboost': {
        'depth': 6,
        'learning_rate': 0.05,
        'iterations': 200,
        'l2_leaf_reg': 3.0,
        'random_state': RANDOM_STATE,
        'verbose': False,
        'thread_count': -1
    },
    'random_forest': {
        'max_depth': 10,            
        'n_estimators': 200,
        'min_samples_split': 10,   
        'min_samples_leaf': 5,      
        'max_features': 'sqrt',
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
}

OPTUNA_N_TRIALS = 100
OPTUNA_TIMEOUT = 7200 

FEATURE_GROUPS = {
    'admission': ['NAM_TUYENSINH', 'PTXT', 'TOHOP_XT', 'DIEM_TRUNGTUYEN', 'DIEM_CHUAN'],
    'academic': ['HOC_KY', 'CPA', 'GPA', 'TC_DANGKY'],
    'engineered': [] 
}

METRICS = ['RMSE']

EARLY_STOPPING_ROUNDS = 50