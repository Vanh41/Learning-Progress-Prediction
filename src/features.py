import pandas as pd
import numpy as np
from .utils import calculate_semester_from_admission, parse_semester_code, fast_slope, get_semester_order


class FeatureEngineer:    
    def __init__(self):
        self.feature_names = ['TOHOP_XT', 'PTXT', 'is_semester_2', 'is_overloaded', 'is_first_semester', 'failed_last_sem']
        
    def create_features(self, df, is_training=True):
        df = df.copy()
        if 'HOC_KY' in df.columns:
            df['year'], df['sem'] = zip(*df['HOC_KY'].apply(parse_semester_code))
            df['semester_order'] = df['year'] * 10 + df['sem']
            df = df.sort_values(['MA_SO_SV', 'semester_order']).reset_index(drop=True)
        grouped = df.groupby('MA_SO_SV')
        df['Prev_GPA'] = grouped['GPA'].shift(1).fillna(-1)
        df['Prev_CPA'] = grouped['CPA'].shift(1).fillna(-1)
        df['Prev_TC_HOANTHANH'] = grouped['TC_HOANTHANH'].shift(1).fillna(0)
        df['Prev_TC_DANGKY'] = grouped['TC_DANGKY'].shift(1).fillna(0)
        df['is_first_semester'] = (df['Prev_TC_DANGKY'] == 0).astype(int)
        df = self._create_admission_features(df)
        df = self._create_history_features(df)
        df = self._create_temporal_features(df)
        df = self._create_trend_features(df)
        df = self._create_academic_features(df)
        # 1. Expected Credits: Dự báo tín chỉ dựa trên tỷ lệ đậu lịch sử
        grouped = df.groupby('MA_SO_SV')
        if 'TC_DANGKY' in df.columns:
            # Tỷ lệ đậu thực tế xuyên suốt lịch sử (ổn định hơn global_pass_rate cũ)
            cum_ht = grouped['Prev_TC_HOANTHANH'].cumsum()
            cum_dk = grouped['Prev_TC_DANGKY'].cumsum()
            df['historical_pass_rate'] = (cum_ht / (cum_dk + 1e-5)).fillna(0.85)
            
            # Dự báo số TC sẽ đạt dựa trên phong độ quá khứ
            df['expected_real_credits'] = df['TC_DANGKY'] * df['historical_pass_rate']
            
            # Risk Interaction: Áp lực kỳ này x Tỷ lệ rớt tích lũy
            df['risk_score'] = df['TC_DANGKY'] * (1 - df['historical_pass_rate'])
        # 5. Đột biến rớt (Recent Fail Spike)
        df['recent_fail_spike'] = df['failed_last_sem'] * (df['TC_DANGKY'] - df['Prev_TC_HOANTHANH'])
        # Dọn dẹp các cột tạm
        df = df.drop(columns=['year', 'sem'], errors='ignore')
        return df
    
    def _create_admission_features(self, df):
        if 'DIEM_TRUNGTUYEN' in df.columns and 'DIEM_CHUAN' in df.columns:
            df['diem_vuot_chuan'] = df['DIEM_TRUNGTUYEN'] - df['DIEM_CHUAN']
            # df['diem_ratio'] = df['DIEM_TRUNGTUYEN'] / (df['DIEM_CHUAN'] + 0.01)
        
        if 'NAM_TUYENSINH' in df.columns:
            df['nam_tuoi'] = 2025 - df['NAM_TUYENSINH']
        
        return df
    
    def _create_academic_features(self, df):        
        if 'TC_DANGKY' in df.columns:
            df['tc_dangky_high'] = (df['TC_DANGKY'] > 25).astype(int)
            df['tc_dangky_low'] = (df['TC_DANGKY'] < 12).astype(int)
        return df
    
    def _create_history_features(self, df):
        df['prev_completion_rate'] = df['Prev_TC_HOANTHANH'] / (df['Prev_TC_DANGKY'] + 1)
        if 'TC_DANGKY' in df.columns:
            avg_capacity = df.groupby('MA_SO_SV')['Prev_TC_HOANTHANH'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            ).fillna(15)

            df['load_factor'] = df['TC_DANGKY'] / (avg_capacity + 1)
            df['is_overloaded'] = (df['load_factor'] > 1.3).astype(int)

            # Đánh dấu sinh viên rớt tín chỉ ở kỳ trước
            df['failed_last_sem'] = (df['Prev_TC_HOANTHANH'] < df['Prev_TC_DANGKY']).astype(int)

            # Phát hiện sinh viên đăng ký nhiều hơn kỳ trước trong khi vừa rớt
            more_credits = (df['TC_DANGKY'] > df['Prev_TC_DANGKY'])
            df['aggressive_recovery'] = (df['failed_last_sem'] & more_credits).astype(int)
        return df

    def _create_temporal_features(self, df):
        if 'HOC_KY' in df.columns and 'NAM_TUYENSINH' in df.columns:
            df['semester_number'] = df.apply(
                lambda row: calculate_semester_from_admission(row['NAM_TUYENSINH'], row['HOC_KY']),
                axis=1
            )
            df['is_semester_2'] = df['HOC_KY'].apply(lambda x: 1 if 'HK2' in str(x) else 0)
        return df
    
    def _create_trend_features(self, df):
        grouped = df.groupby('MA_SO_SV')
        if 'Prev_GPA' in df.columns:
            # Xu hướng điểm số (Slope)
            df['gpa_trend_slope'] = grouped['Prev_GPA'].transform(
                lambda x: x.rolling(window=3, min_periods=2).apply(fast_slope, raw=True)
            ).fillna(0)
        if 'Prev_TC_DANGKY' in df.columns:
            cum_dangky = grouped['Prev_TC_DANGKY'].cumsum()
            cum_hoanthanh = grouped['Prev_TC_HOANTHANH'].cumsum()
            df['total_credits_failed'] = cum_dangky - cum_hoanthanh
            df['accumulated_fail_ratio'] = df['total_credits_failed'] / (cum_dangky + 1e-5)
        return df
    
    def create_test_features(self, test_df, train_df, test_semester='HK1 2024-2025'):
        """
        Tạo feature cho tập Test bằng cách nối với lịch sử để tính toán các biến lũy kế.
        """
        print(f"Creating test features for {test_semester}...")

        # 1. Chuẩn bị tập test
        test_df = test_df.copy()
        test_df['HOC_KY'] = test_semester
        
        # Giả định tập test chưa có GPA, CPA, TC_HOANTHANH (đối tượng cần dự báo)
        # Ta gán nhãn tạm để có thể concat với train_df
        for col in ['GPA', 'CPA', 'TC_HOANTHANH']:
            if col not in test_df.columns:
                test_df[col] = 0

        # 2. Nối với lịch sử (train_df) để tính toán các biến trễ (Lag) và lũy kế (Cumulative)
        full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

        # 3. Chạy toàn bộ pipeline tạo feature
        full_feat = self.create_features(full_df)

        # 4. Trích xuất đúng các dòng thuộc về học kỳ test
        test_order = get_semester_order(test_semester)
        test_features = full_feat[full_feat['semester_order'] == test_order].copy()

        # 5. Loại bỏ các cột target tạm thời
        cols_to_drop = ['TC_HOANTHANH', 'GPA', 'CPA']
        test_features = test_features.drop(columns=[c for c in cols_to_drop if c in test_features.columns])

        print(f"Test features created: {test_features.shape}")
        return test_features

    
    def get_feature_columns(self, df, exclude_cols=None):
        valid_prefixes = [
            'Prev_', 'prev_', 'sem_', 'diem_', 'nam_', 'is_',
            'load_', 'aggressive_', 'gpa_trend', 'total_', 'accumulated_',
            'expected_', 'historical_', 'risk_', 'failed_', 'recent_', 'tc_dangky_'
        ]

        valid_exact = ['TC_DANGKY', 'NAM_TUYENSINH', 'DIEM_TRUNGTUYEN', 'DIEM_CHUAN', 'semester_number', 'TOHOP_XT', 'PTXT']

        # Các cột không được đưa vào feature list
        exclude_cols = ['TC_HOANTHANH', 'GPA', 'CPA', 'semester_order', 'MA_SO_SV', 'year', 'sem', 'HOC_KY']

        return [c for c in df.columns if c not in exclude_cols and 
                (c in valid_exact or any(c.startswith(p) for p in valid_prefixes))]


def prepare_features_for_modeling(train_df, valid_df, target_col='TC_HOANTHANH'):
    engineer = FeatureEngineer()

    print("Creating train + valid features (time-aware)...")

    # 1. GỘP để tạo lịch sử liên tục
    full_df = pd.concat([train_df, valid_df], axis=0, ignore_index=True)

    full_df = engineer.create_features(full_df)

    # 2. TÁCH LẠI
    train_end_order = get_semester_order('HK1 2023-2024')   # 20231
    valid_order = get_semester_order('HK2 2023-2024')       # 20232
    train_feat = full_df[full_df['semester_order'] <= train_end_order].copy()
    valid_feat = full_df[full_df['semester_order'] == valid_order].copy()

    # 4. Lấy danh sách feature
    feature_cols = engineer.get_feature_columns(train_feat)

    X_train = train_feat[feature_cols]
    y_train = train_feat[target_col]

    X_valid = valid_feat[feature_cols]
    y_valid = valid_feat[target_col]

    print(f"\nFeature validation:")
    print(f"  Number of features   : {len(feature_cols)}")
    categorical_cols = [c for c in feature_cols if c in engineer.feature_names]
    print(f"  Categorical features : {categorical_cols}")
    print(f"  Train size           : {X_train.shape}")
    print(f"  Valid size           : {X_valid.shape}")

    return X_train, X_valid, y_train, y_valid, feature_cols, categorical_cols