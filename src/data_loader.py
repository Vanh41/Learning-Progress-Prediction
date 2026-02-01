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
        self.admission_df = self._clean_admission_data(self.admission_df)
        self.academic_df = self._clean_academic_data(self.academic_df)
        return self
    
    def _clean_admission_data(self, df):
        """Clean admission data"""
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['MA_SO_SV'])
        
        # Fill missing categorical values
        if 'PTXT' in df.columns:
            df['PTXT'] = df['PTXT'].fillna('Unknown')
    
        if 'TOHOP_XT' in df.columns:
            df['TOHOP_XT'] = df['TOHOP_XT'].fillna('Unknown')
        
        # Fill missing numeric values with median
        for col in ['DIEM_TRUNGTUYEN', 'DIEM_CHUAN']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
                
        return df
    
    def _clean_academic_data(self, df):
        """Clean academic data"""
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['MA_SO_SV', 'HOC_KY'])
        
        # Fill missing GPA/CPA with 0
        for col in ['GPA', 'CPA']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Fill missing TC values
        for col in ['TC_DANGKY', 'TC_HOANTHANH']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Ensure TC_HOANTHANH <= TC_DANGKY (data validation)
        if 'TC_HOANTHANH' in df.columns and 'TC_DANGKY' in df.columns:
            df['TC_HOANTHANH'] = df[['TC_HOANTHANH', 'TC_DANGKY']].min(axis=1)
        
        # Ensure non-negative values
        for col in ['GPA', 'CPA', 'TC_DANGKY', 'TC_HOANTHANH']:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
        
        return df
    
    def merge_data(self):
        """Merge academic records with admission data"""
        print("Merging data...")
        
        merged_df = self.academic_df.merge(
            self.admission_df,
            on='MA_SO_SV',
            how='left'
        )
        
        print(f"Merged data shape: {merged_df.shape}")
        
        # Validate merge
        if merged_df.isnull().any().any():
            print("Warning: Found null values after merge")
            null_counts = merged_df.isnull().sum()
            print(null_counts[null_counts > 0])
        
        return merged_df
    
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