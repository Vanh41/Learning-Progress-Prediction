import os
import random
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def save_model(model, filename, directory='models'):
    from .config import MODELS_DIR
    
    filepath = MODELS_DIR / filename
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")
    return str(filepath)


def load_model(filename, directory='models'):
    from .config import MODELS_DIR
    
    filepath = MODELS_DIR / filename
    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model


def save_submission(predictions, student_ids, team_name, directory='output'):
    from .config import OUTPUT_DIR
    
    submission = pd.DataFrame({
        'MA_SO_SV': student_ids,
        'PRED_TC_HOANTHANH': predictions.astype(int)
    })
    
    filename = f"{team_name}.csv"
    filepath = OUTPUT_DIR / filename
    submission.to_csv(filepath, index=False)
    print(f"Submission saved to: {filepath}")
    return str(filepath)


def create_semester_code(year, semester):
    next_year = year + 1
    return f"HK{semester} {year}-{next_year}"


def parse_semester_code(semester_code):
    parts = semester_code.strip().split()
    semester = int(parts[0].replace('HK', ''))
    year_range = parts[1].split('-')
    year = int(year_range[0])
    return year, semester


def get_semester_order(semester_code):
    year, semester = parse_semester_code(semester_code)
    return year * 10 + semester


def calculate_semester_from_admission(admission_year, current_semester_code):
    current_year, current_sem = parse_semester_code(current_semester_code)
    years_diff = current_year - admission_year
    return years_diff * 2 + current_sem


def log_experiment(experiment_name, metrics, params, directory='output'):
    from .config import OUTPUT_DIR
    import pandas as pd
    from datetime import datetime

    log_file = OUTPUT_DIR / 'experiment_log.csv'

    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experiment': experiment_name,
        **metrics,
        **{f'param_{k}': v for k, v in params.items()}
    }

    if log_file.exists() and log_file.stat().st_size > 0:
        log_df = pd.read_csv(log_file)
        log_df = pd.concat(
            [log_df, pd.DataFrame([log_entry])],
            ignore_index=True
        )
    else:
        log_df = pd.DataFrame([log_entry])

    log_df.to_csv(log_file, index=False)
    print(f"Experiment logged to: {log_file}")

    return log_file

def memory_usage(df):
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    return f"{memory_mb:.2f} MB"