import os
import random
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
import re

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def save_model(model, filename, directory='models'):
    from pathlib import Path
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    filepath = models_dir / filename
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")
    return str(filepath)


def load_model(filename, directory='models'):
    from pathlib import Path
    filepath = Path('models') / filename
    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model


def save_submission(predictions, student_ids, team_name, directory='output'):
    from pathlib import Path
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    submission = pd.DataFrame({
        'MA_SO_SV': student_ids,
        'PRED_TC_HOANTHANH': predictions
    })
    
    filename = f"{team_name}_submission.csv"
    filepath = output_dir / filename
    submission.to_csv(filepath, index=False)
    print(f"Submission saved to: {filepath}")
    return str(filepath)


def create_semester_code(year, semester):
    next_year = year + 1
    return f"HK{semester} {year}-{next_year}"


def parse_semester_string(sem_str):
    """Tách học kỳ và năm học từ chuỗi (VD: 'HK1 2023-2024')"""
    s = str(sem_str).strip()
    if s.isdigit():
        return int(s)
    digits=re.findall(r'\d+',s)
    if len(digits) >= 2:
        # Giả sử số nhỏ là kỳ, số lớn (4 chữ số) là năm
        # Tìm năm (thường là số có 4 chữ số đầu tiên tìm thấy)
        years = [int(d) for d in digits if len(d) == 4]
        sems = [int(d) for d in digits if len(d) == 1]
        
        if years and sems:
            year = years[0]
            sem = sems[0]
            return year * 10 + sem
    return 0

def parse_semester_code(semester_code):
    if pd.isna(semester_code):
        return 0, 0
    try:
        parts = str(semester_code).strip().split()
        semester = int(parts[0].replace('HK', ''))
        year_part = parts[1].split('-')[0]
        year = int(year_part)
        return year, semester
    except:
        return 0, 0

def get_semester_order(semester_code):
    return parse_semester_string(semester_code)


def calculate_semester_from_admission(admission_year, semester_code):
    """Tính sinh viên đang học ở kỳ thứ mấy"""
    if pd.isna(admission_year) or pd.isna(semester_code):
        return 0
    current_year, current_sem = parse_semester_code(semester_code)
    if current_year == 0:
        return 0
    semester_num = (current_year - admission_year) * 2 + current_sem
    return max(1, semester_num)

def fast_slope(y):
    y_clean = y[~np.isnan(y)]
    n = len(y_clean)
    if n < 2:
        return 0.0
    
    x = np.arange(n)
    x_mean = np.mean(x)
    y_mean = np.mean(y_clean)
    
    numerator = np.sum((x - x_mean) * (y_clean - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    return numerator / (denominator + 1e-6)

def log_experiment(experiment_name, metrics, params):
    """Log experiment results"""
    from pathlib import Path
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    log_file = output_dir / 'experiment_log.csv'

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
