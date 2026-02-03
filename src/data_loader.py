import pandas as pd
import numpy as np
from pathlib import Path
from .config import ADMISSION_FILE, ACADEMIC_RECORDS_FILE, TEST_FILE
from .utils import get_semester_order, parse_semester_code
import re

class DataLoader:
    def __init__(self, admission_path, academic_path, test_path=None):
        self.admission_path = admission_path
        self.academic_path = academic_path
        self.test_path = test_path
        self.admission_df = None
        self.academic_df = None
        self.test_df = None
        self.merged_df = None
        
    def load_raw_data(self):
        """Load raw data from CSV files"""
        print("Loading raw data...")
        self.admission_df = pd.read_csv(self.admission_path)
        print(f"Admission data loaded: {self.admission_df.shape}")
        
        self.academic_df = pd.read_csv(self.academic_path)
        print(f"Academic records loaded: {self.academic_df.shape}")
        
        if self.test_path and Path(self.test_path).exists():
            self.test_df = pd.read_csv(self.test_path)
            print(f"Test data loaded: {self.test_df.shape}")
        else:
            print("Test file not found or not provided")
        
        return self
    
    def parse_semester_string(self, sem_str):
        """Chuyển đổi 'HK1 2023-2024' -> 20231 (Int) để sort time-series"""
        s = str(sem_str).strip()
        # Trường hợp 1: Dạng số sẵn (VD: 20231)
        if s.isdigit():
            return int(s)
        # Trường hợp 2: Dạng chữ (VD: HK1 2023-2024)
        digits = re.findall(r'\d+', s)
        if len(digits) >= 2:
            years = [int(d) for d in digits if len(d) == 4]
            sems = [int(d) for d in digits if len(d) == 1]
            if years and sems:
                return years[0] * 10 + sems[0]
                
        return 0
    
    def clean_data(self, is_test=False):
        """Clean and preprocess raw data"""
        print(f"--- PREPROCESSING DATA (is_test={is_test}) ---")
        adm = self.admission_df.copy()
        acad = self.academic_df.copy()
        
        # Chuẩn hóa ID
        adm['MA_SO_SV'] = adm['MA_SO_SV'].astype(str)
        acad['MA_SO_SV'] = acad['MA_SO_SV'].astype(str)
        
        # Tạo Time-Index
        print("-> Đang xử lý cột HOC_KY...")
        acad['semester_order'] = acad['HOC_KY'].apply(self.parse_semester_string)
        
        # Kiểm tra xem có dòng nào bị lỗi (bằng 0) không
        error_count = (acad['semester_order'] == 0).sum()
        if error_count > 0:
            print(f"   Cảnh báo: Có {error_count} dòng không đọc được HOC_KY.")
        
        # Merge
        df = pd.merge(acad, adm, on='MA_SO_SV', how='left')
        
        # Sort Time-Series (CỰC KỲ QUAN TRỌNG)
        df = df.sort_values(by=['MA_SO_SV', 'semester_order']).reset_index(drop=True)
        
        # Numeric conversion
        cols_float = ['GPA', 'CPA', 'DIEM_TRUNGTUYEN', 'DIEM_CHUAN']
        cols_int = ['TC_DANGKY', 'TC_HOANTHANH']
        
        for col in cols_float:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        for col in cols_int:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Logic clean (chỉ cho training data)
        if not is_test:
            initial_len = len(df)
            # Logic: Hoàn thành <= Đăng ký
            df['TC_HOANTHANH'] = np.minimum(df['TC_HOANTHANH'], df['TC_DANGKY'])
            # Clip điểm số
            df['GPA'] = df['GPA'].clip(0, 4.0)
            df['CPA'] = df['CPA'].clip(0, 4.0)
            # Lọc rác (TC_DANGKY > 0)
            df = df[df['TC_DANGKY'] > 0].copy()
            dropped = initial_len - len(df)
            if dropped > 0:
                print(f" -> Đã loại bỏ {dropped} dòng rác (TC_DANGKY=0).")
            # Target Transformation
            df['COMPLETION_RATE'] = df['TC_HOANTHANH'] / (df['TC_DANGKY'] + 1e-9)
            df['COMPLETION_RATE'] = df['COMPLETION_RATE'].clip(0, 1)
        
        # Admission Gap Feature
        if 'DIEM_TRUNGTUYEN' in df.columns and 'DIEM_CHUAN' in df.columns:
            df['ADMISSION_GAP'] = df['DIEM_TRUNGTUYEN'] - df['DIEM_CHUAN']
        
        print(f"--- HOÀN TẤT. Kích thước data: {df.shape} ---")
        if 'semester_order' in df.columns:
            print("Sample semester_order:", df['semester_order'].head().tolist())
        
        self.merged_df = df
        return self
    
    def prepare_test_data(self, test_semester='HK1 2024-2025'):
        if self.test_df is None:
            print("No test data loaded")
            return None
        print(f"Preparing test data for {test_semester}...")
        test_copy = self.test_df.copy()
        test_copy['HOC_KY'] = test_semester
        for col in ['TC_HOANTHANH', 'GPA', 'CPA']:
            if col not in test_copy.columns:
                test_copy[col] = 0
        original_academic = self.academic_df
        self.academic_df = test_copy
        self.clean_data(is_test=True)
        test_processed = self.merged_df.copy()
        self.academic_df = original_academic
        self.clean_data(is_test=False)  
        return test_processed
    
    def merge_train_test_for_features(self, test_df):
        if test_df is None:
            return self.merged_df, None
        train_df = self.merged_df.copy()
        train_df['set_type'] = 'TRAIN'
        test_df = test_df.copy()
        test_df['set_type'] = 'TEST'
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        return full_df
    
    def split_train_valid(self, df, split_sem=20231, valid_sem=20232):
        print(f"\nSplitting data: Train <= {split_sem}, Valid = {valid_sem}")
        train_mask = df['semester_order'] <= split_sem
        valid_mask = df['semester_order'] == valid_sem
        train_df = df[train_mask].copy()
        valid_df = df[valid_mask].copy()
        print(f"Train size: {train_df.shape}")
        print(f"Valid size: {valid_df.shape}")
        return train_df, valid_df
    
    def get_merged_data(self):
        return self.merged_df
    
    def get_test_data(self):
        return self.test_df


def load_and_prepare_data(admission_path, academic_path, test_path=None, 
                         split_sem=20231, valid_sem=20232):
    loader = DataLoader(admission_path, academic_path, test_path)
    loader.load_raw_data()
    loader.clean_data(is_test=False)
    test_processed = None
    if test_path and Path(test_path).exists():
        test_processed = loader.prepare_test_data()
    merged_df = loader.get_merged_data()
    train_df, valid_df = loader.split_train_valid(merged_df, split_sem, valid_sem)
    return train_df, valid_df, test_processed