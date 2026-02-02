import pandas as pd
import numpy as np
from pathlib import Path
from .config import ADMISSION_FILE, ACADEMIC_RECORDS_FILE, TEST_FILE
from .utils import get_semester_order, parse_semester_code


class DataLoader:
    def __init__(self):
        self.admission_df = None
        self.academic_df = None
        self.test_df = None
        
    def load_raw_data(self):
        """Load raw data from CSV files"""
        print("Loading raw data...")
        self.admission_df = pd.read_csv(ADMISSION_FILE)
        print(f"Admission data loaded: {self.admission_df.shape}")
        
        self.academic_df = pd.read_csv(ACADEMIC_RECORDS_FILE)
        print(f"Academic records loaded: {self.academic_df.shape}")
        
        if TEST_FILE.exists():
            self.test_df = pd.read_csv(TEST_FILE)
            print(f"Test data loaded: {self.test_df.shape}")
        else:
            print("Test file not found")
        
        return self
    
    def clean_data(self):
        """Clean and preprocess raw data"""
        print("Cleaning data...")
        self.admission_df['MA_SO_SV'] = self.admission_df['MA_SO_SV'].astype(str)
        self.academic_df['MA_SO_SV'] = self.academic_df['MA_SO_SV'].astype(str)
        df = pd.merge(self.academic_df, self.admission_df, on='MA_SO_SV', how='inner')
        numeric_floats = ['GPA', 'CPA', 'DIEM_TRUNGTUYEN', 'DIEM_CHUAN']
        for col in numeric_floats:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        numeric_ints = ['TC_DANGKY', 'TC_HOANTHANH']
        for col in numeric_ints:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        df = df.sort_values(by=['MA_SO_SV', 'HOC_KY']).reset_index(drop=True)
        # BƯỚC 2: KIỂM TRA LOGIC & LÀM SẠCH NHIỄU (Sanity Checks)
        initial_rows = len(df)
        df['TC_HOANTHANH'] = np.minimum(df['TC_HOANTHANH'], df['TC_DANGKY'])
        df['GPA'] = df['GPA'].clip(lower=0.0, upper=4.0)
        df['CPA'] = df['CPA'].clip(lower=0.0, upper=4.0)
        rows_before_score_filter = len(df)
        df = df[df['DIEM_TRUNGTUYEN'] >= df['DIEM_CHUAN']]
        dropped_score_rows = rows_before_score_filter - len(df)
        print(f" -> Đã loại bỏ {dropped_score_rows} dòng do Điểm trúng tuyển < Điểm chuẩn.")
        rows_before_credit_filter = len(df)
        df = df[df['TC_DANGKY'] > 0].copy()
        dropped_credit_rows = rows_before_credit_filter - len(df)
        print(f" -> Đã loại bỏ {dropped_credit_rows} dòng rác (TC_DANGKY=0).")
        total_dropped = initial_rows - len(df)
        print(f"--- HOÀN TẤT: Tổng cộng đã loại bỏ {total_dropped} dòng nhiễu. Kích thước data cuối: {df.shape} ---")
        
        self.merged_df = df
        return self
    
    def merge_data(self):
        """Return merged and cleaned data"""
        print(f"Merged data shape: {self.merged_df.shape}")
        return self.merged_df
    
    def split_data(self, merged_df, train_end='HK1 2023-2024', valid_semester='HK2 2023-2024'):
        """Split data into train and validation sets based on semester"""
        print("Splitting data into train and validation sets...")
        
        # Add semester order for splitting
        merged_df['semester_order'] = merged_df['HOC_KY'].apply(get_semester_order)
        
        train_end_order = get_semester_order(train_end)
        valid_order = get_semester_order(valid_semester)
        
        # Split based on semester
        train_df = merged_df[merged_df['semester_order'] <= train_end_order].copy()
        valid_df = merged_df[merged_df['semester_order'] == valid_order].copy()
        
        print(f"Train data: {train_df.shape} (semesters <= {train_end})")
        print(f"Valid data: {valid_df.shape} (semester = {valid_semester})")
        
        # Validate split
        if len(train_df) == 0:
            raise ValueError("Training set is empty!")
        if len(valid_df) == 0:
            print("Warning: Validation set is empty!")
        
        # Check for data leakage
        train_students = set(train_df['MA_SO_SV'].unique())
        valid_students = set(valid_df['MA_SO_SV'].unique())
        print(f"Train students: {len(train_students)}")
        print(f"Valid students: {len(valid_students)}")
        print(f"Overlap: {len(train_students & valid_students)} students")
        
        return train_df, valid_df
    
    def get_test_data(self):
        """Get test data"""
        return self.test_df
    
    def save_processed_data(self, train_df, valid_df, directory='processed'):
        """Save processed data to disk"""
        from .config import PROCESSED_DATA_DIR
        
        train_path = PROCESSED_DATA_DIR / 'train_data.csv'
        valid_path = PROCESSED_DATA_DIR / 'valid_data.csv'
        
        train_df.to_csv(train_path, index=False)
        valid_df.to_csv(valid_path, index=False)
        
        print(f"Processed data saved to {PROCESSED_DATA_DIR}")
        
        return train_path, valid_path


def load_and_prepare_data(train_end='HK1 2023-2024', valid_semester='HK2 2023-2024'):
    """One-liner to load and prepare all data"""
    loader = DataLoader()
    loader.load_raw_data()
    loader.clean_data()
    merged_df = loader.merge_data()
    train_df, valid_df = loader.split_data(merged_df, train_end, valid_semester)
    test_df = loader.get_test_data()
    
    return train_df, valid_df, test_df