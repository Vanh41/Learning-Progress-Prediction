import pandas as pd
import numpy as np
from src.utils import fast_slope, parse_semester_string, calculate_semester_from_admission
from sklearn.preprocessing import OrdinalEncoder

class FeatureEngineer:    
    def __init__(self):
        self.cat_cols = ['PTXT', 'TOHOP_XT', 'MA_NGANH', 'KV_UT', 'KHOA_VIEN'] 
        
    def create_features(self, df):
        print("--- FEATURE ENGINEERING ---")
        df = df.copy()
        df = df.sort_values(['MA_SO_SV', 'semester_order']).reset_index(drop=True)
        for col in self.cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).astype('category')
        g = df.groupby('MA_SO_SV')
        df['Prev_GPA_Raw'] = g['GPA'].shift(1)
        df['Prev_GPA'] = df['Prev_GPA_Raw'].fillna(-1)
        df['Prev_CPA'] = g['CPA'].shift(1).fillna(-1)
        df['Prev_TC_HOANTHANH'] = g['TC_HOANTHANH'].shift(1).fillna(0)
        df['Prev_TC_DANGKY'] = g['TC_DANGKY'].shift(1).fillna(0)
        df['is_freshman'] = (df['Prev_TC_DANGKY'] == 0).astype(int)
        df = self._create_admission_features(df)
        df = self._create_history_features(df)
        df = self._create_trend_features(df)  # <--- Slope, Volatility nằm ở đây
        df = self._create_risk_features(df)
        
        if 'Prev_GPA_Raw' in df.columns:
            df = df.drop(columns=['Prev_GPA_Raw'])
            
        print(f" Feature engineering completed. Shape: {df.shape}")
        return df
    
    def _create_admission_features(self, df):
        if 'DIEM_TRUNGTUYEN' in df.columns and 'DIEM_CHUAN' in df.columns:
            df['diem_vuot_chuan'] = df['DIEM_TRUNGTUYEN'] - df['DIEM_CHUAN']
            # df['diem_ratio'] = df['DIEM_TRUNGTUYEN'] / (df['DIEM_CHUAN'] + 0.01)
        
        if 'NAM_TUYENSINH' in df.columns:
            df['nam_tuoi'] = 2026 - df['NAM_TUYENSINH']
        df['semester_number'] = df.groupby('MA_SO_SV').cumcount() + 1
        return df
    
    def _create_history_features(self, df):
        df['prev_gpa_cpa_diff'] = df['Prev_GPA'] - df['Prev_CPA']
        df['prev_completion_rate'] = df['Prev_TC_HOANTHANH'] / (df['Prev_TC_DANGKY'] + 1e-9)
        avg_capacity = df.groupby('MA_SO_SV')['Prev_TC_HOANTHANH'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        ).fillna(15)
        df['load_factor'] = df['TC_DANGKY'] / (avg_capacity + 1e-9)
        df['failed_last_sem'] = (df['Prev_TC_HOANTHANH'] < df['Prev_TC_DANGKY']).astype(int)
        
        return df
    
    def _create_trend_features(self, df):
        g_raw = df.groupby('MA_SO_SV')['Prev_GPA_Raw']
        df['gpa_trend_slope'] = g_raw.transform(
            lambda x: x.rolling(window=3, min_periods=2).apply(fast_slope, raw=True)
        ).fillna(0)
        df['gpa_volatility'] = g_raw.transform(
            lambda x: x.rolling(window=4, min_periods=2).std()
        ).fillna(0)
        grouped = df.groupby('MA_SO_SV')
        cum_dangky = grouped['Prev_TC_DANGKY'].cumsum()
        cum_hoanthanh = grouped['Prev_TC_HOANTHANH'].cumsum()
        df['total_credits_failed'] = cum_dangky - cum_hoanthanh
        df['accumulated_fail_ratio'] = df['total_credits_failed'] / (cum_dangky + 1e-9)
        semester_count = grouped.cumcount() + 1
        df['credit_velocity'] = cum_hoanthanh / semester_count
        return df
    
    def _create_risk_features(self, df):
        more_credits = (df['TC_DANGKY'] > df['Prev_TC_DANGKY'])
        df['aggressive_recovery'] = (df['failed_last_sem'] & more_credits).astype(int)
        df['expected_real_credits'] = df['TC_DANGKY'] * (1 - df['accumulated_fail_ratio'])
        return df
    
    def get_feature_columns(self, df):
        valid_prefixes = [
            'Prev_', 'prev_', 'sem_', 'diem_', 'nam_', 'is_', 
            'load_', 'aggressive_', 'gpa_', 'total_', 'accumulated_',
            'credit_', 'expected_', 'failed_'
        ]

        valid_exact = ['TC_DANGKY', 'DIEM_TRUNGTUYEN', 'DIEM_CHUAN', 'semester_number']
        valid_exact.extend(self.cat_cols)
        # Các cột không được đưa vào feature list
        final_cols = []
        ignore_cols = [
            'TC_HOANTHANH', 'GPA', 'CPA', 'semester_order', 
            'MA_SO_SV', 'HOC_KY', 'COMPLETION_RATE', 'Prev_GPA_Raw',
            'set_type', 'ADMISSION_GAP'  # set_type là cột temporary
        ]
        for col in df.columns:
            if col in ignore_cols:
                continue
            
            is_valid = False
            if col in valid_exact:
                is_valid = True
            else:
                for prefix in valid_prefixes:
                    if col.startswith(prefix):
                        is_valid = True
                        break
            
            if is_valid:
                final_cols.append(col)
        
        return final_cols

def prepare_features_for_modeling(train_df, valid_df, test_df=None, target_col='TC_HOANTHANH'):
    engineer = FeatureEngineer()

    print(" FULL FEATURE ENGINEERING PIPELINE")

    # 1. Gộp train + valid (+ test nếu có) để tạo features liên tục
    dfs_to_concat = []
    
    train_copy = train_df.copy()
    train_copy['set_type'] = 'TRAIN'
    dfs_to_concat.append(train_copy)
    
    valid_copy = valid_df.copy()
    valid_copy['set_type'] = 'VALID'
    dfs_to_concat.append(valid_copy)
    
    if test_df is not None:
        test_copy = test_df.copy()
        test_copy['set_type'] = 'TEST'
        dfs_to_concat.append(test_copy)
    
    full_df = pd.concat(dfs_to_concat, ignore_index=True)

    # 2. Feature Engineering trên toàn bộ data
    full_df_fe = engineer.create_features(full_df)

    # 3. Tách lại Train/Valid/Test
    train_feat = full_df_fe[full_df_fe['set_type'] == 'TRAIN'].copy()
    valid_feat = full_df_fe[full_df_fe['set_type'] == 'VALID'].copy()
    
    # 4. Lấy danh sách feature
    feature_cols = engineer.get_feature_columns(train_feat)
    categorical_cols = [c for c in feature_cols if c in engineer.cat_cols]

    # 5. Prepare X, y
    X_train = train_feat[feature_cols].copy()
    y_train = train_feat[target_col] if target_col in train_feat.columns else None

    X_valid = valid_feat[feature_cols].copy()
    y_valid = valid_feat[target_col] if target_col in valid_feat.columns else None
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1)
    if categorical_cols:
        X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
        X_valid[categorical_cols] = encoder.transform(X_valid[categorical_cols])

    print(f"\n Feature preparation complete:")
    print(f"  Number of features   : {len(feature_cols)}")
    print(f"  Categorical features : {categorical_cols}")
    print(f"  Train size           : {X_train.shape}")
    print(f"  Valid size           : {X_valid.shape}")

    # 6. Nếu có test data
    result = {
        'X_train': X_train,
        'X_valid': X_valid,
        'y_train': y_train,
        'y_valid': y_valid,
        'feature_cols': feature_cols,
        'categorical_cols': categorical_cols
    }
    
    if test_df is not None:
        test_feat = full_df_fe[full_df_fe['set_type'] == 'TEST'].copy()
        X_test = test_feat[feature_cols].copy()
        X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])
        result['X_test'] = X_test
        result['test_df'] = test_feat
        print(f"  Test size            : {X_test.shape}")

    return result