import pandas as pd
import numpy as np
from scipy.stats import linregress  
from .utils import calculate_semester_from_admission, parse_semester_code

# Thêm hàm này vào trước class FeatureEngineer
def calculate_slope(series):
    y = series.dropna()
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    # Tính nhanh hệ số góc
    slope, _, _, _, _ = linregress(x, y)
    return slope

class FeatureEngineer:    
    def __init__(self):
        self.feature_names = []
        
    def create_features(self, df, is_training=True):
        df = df.copy()
        df = self._create_admission_features(df)
        df = self._create_academic_features(df)
        df = self._create_temporal_features(df)
        df = self._create_performance_features(df)
        df = self._create_aggregated_features(df)
        df = self._create_creative_features(df)
        df = self._create_advanced_features(df)
        # [NEW] Nhóm biến xu hướng & so sánh
        # df = self._create_trend_features(df)
        # df = self._create_cohort_features(df)
        
        return df
    
    def _create_admission_features(self, df):
        if 'DIEM_TRUNGTUYEN' in df.columns and 'DIEM_CHUAN' in df.columns:
            df['diem_vuot_chuan'] = df['DIEM_TRUNGTUYEN'] - df['DIEM_CHUAN']
            df['diem_ratio'] = df['DIEM_TRUNGTUYEN'] / (df['DIEM_CHUAN'] + 0.01)
        
        if 'NAM_TUYENSINH' in df.columns:
            df['nam_tuoi'] = 2024 - df['NAM_TUYENSINH']
        
        return df
    
    def _create_academic_features(self, df):        
        if 'GPA' in df.columns and 'CPA' in df.columns:
            df['gpa_cpa_diff'] = df['GPA'] - df['CPA']
            df['gpa_cpa_ratio'] = df['GPA'] / (df['CPA'] + 0.01)
        
        if 'TC_DANGKY' in df.columns:
            df['tc_dangky_high'] = (df['TC_DANGKY'] > 20).astype(int)
            df['tc_dangky_low'] = (df['TC_DANGKY'] < 15).astype(int)
        
        if 'TC_HOANTHANH' in df.columns and 'TC_DANGKY' in df.columns:
            df['completion_rate'] = df['TC_HOANTHANH'] / (df['TC_DANGKY'] + 0.01)
            df['tc_failed'] = df['TC_DANGKY'] - df['TC_HOANTHANH']
        
        return df
    
    def _create_temporal_features(self, df):        
        if 'HOC_KY' in df.columns and 'NAM_TUYENSINH' in df.columns:
            df['semester_number'] = df.apply(
                lambda row: calculate_semester_from_admission(
                    row['NAM_TUYENSINH'], 
                    row['HOC_KY']
                ) if pd.notna(row['HOC_KY']) else 0,
                axis=1
            )
            
            df['hoc_ky_nam'] = df['HOC_KY'].apply(
                lambda x: parse_semester_code(x)[0] if pd.notna(x) else 0
            )
            df['hoc_ky_so'] = df['HOC_KY'].apply(
                lambda x: parse_semester_code(x)[1] if pd.notna(x) else 0
            )
            
            df['is_semester_2'] = (df['hoc_ky_so'] == 2).astype(int)
            
        return df
    
    def _create_performance_features(self, df):
        if 'CPA' in df.columns:
            df['cpa_excellent'] = (df['CPA'] >= 3.6).astype(int)
            df['cpa_good'] = ((df['CPA'] >= 3.2) & (df['CPA'] < 3.6)).astype(int)
            df['cpa_fair'] = ((df['CPA'] >= 2.5) & (df['CPA'] < 3.2)).astype(int)
            df['cpa_poor'] = (df['CPA'] < 2.5).astype(int)
        
        if 'GPA' in df.columns:
            df['gpa_excellent'] = (df['GPA'] >= 3.6).astype(int)
            df['gpa_good'] = ((df['GPA'] >= 3.2) & (df['GPA'] < 3.6)).astype(int)
            df['gpa_fair'] = ((df['GPA'] >= 2.5) & (df['GPA'] < 3.2)).astype(int)
            df['gpa_poor'] = (df['GPA'] < 2.5).astype(int)
        
        return df
    
    def _create_creative_features(self, df):
        """
        Tạo các biến đặc thù (Creative Features) để bắt hành vi và tâm lý sinh viên.
        """
        # 1. BIẾN SỨC Ì HỌC THUẬT (GPA Efficiency)
        # Sinh viên đạt GPA cao trên nền tải trọng nặng (đăng ký nhiều tín) -> Năng lực tốt
        if 'GPA' in df.columns and 'TC_DANGKY' in df.columns and 'HOC_KY' in df.columns:
            # Tính trung bình tín chỉ đăng ký của toàn khóa trong kỳ đó (làm mẫu số)
            # Lưu ý: transform('mean') sẽ tính mean của nhóm HOC_KY gán cho từng dòng
            mean_reg_by_sem = df.groupby('HOC_KY')['TC_DANGKY'].transform('mean')
            
            # Chỉ số hiệu quả = GPA / Tỷ lệ tải trọng (tránh chia cho 0)
            # Tải trọng > 1 nghĩa là đăng ký nhiều hơn trung bình
            load_ratio = df['TC_DANGKY'] / (mean_reg_by_sem + 1e-5)
            df['gpa_efficiency_index'] = df['GPA'] * load_ratio

        # 2. BIẾN "CÚ SỐC ĐẦU VÀO" (Admission Shock - Career Misalignment)
        # Kiểm tra xem sinh viên thi khối Xã hội (C, D) nhưng học ngành Kỹ thuật (ví dụ)
        if 'TOHOP_XT' in df.columns:
            # Giả định các khối Tự nhiên gốc
            natural_blocks = ['A00', 'A01', 'B00', 'B08']
            # Tạo cờ: 1 nếu thi khối tự nhiên, 0 nếu khối khác
            df['is_natural_science_entry'] = df['TOHOP_XT'].isin(natural_blocks).astype(int)

        # 3. BIẾN PHẢN ỨNG SAU THẤT BẠI (Post-Failure Reaction)
        # Xem sinh viên làm gì sau khi rớt môn?
        if 'TC_DANGKY' in df.columns and 'TC_HOANTHANH' in df.columns:
            # Tính lượng thay đổi tín chỉ đăng ký so với kỳ trước
            df['prev_tc_dangky'] = df.groupby('MA_SO_SV')['TC_DANGKY'].shift(1).fillna(0)
            df['tc_reg_delta'] = df['TC_DANGKY'] - df['prev_tc_dangky']
            
            # Kiểm tra kỳ trước có rớt môn không
            df['prev_tc_hoanthanh'] = df.groupby('MA_SO_SV')['TC_HOANTHANH'].shift(1).fillna(0)
            df['had_failure_last_sem'] = (df['prev_tc_hoanthanh'] < df['prev_tc_dangky']).astype(int)
            
            # Flag "Nóng vội": Kỳ trước rớt môn + Kỳ này tăng mạnh số tín chỉ (>3 tín)
            # Đây là hành vi rủi ro cao (cố học trả nợ nhưng dễ gãy tiếp)
            df['aggressive_recovery_flag'] = ((df['had_failure_last_sem'] == 1) & 
                                            (df['tc_reg_delta'] > 3)).astype(int)

        # 4. BIẾN ĐÚNG TIẾN ĐỘ (On-Track Proxy)
        # So sánh năm học hiện tại với năm tuyển sinh
        if 'HOC_KY' in df.columns and 'NAM_TUYENSINH' in df.columns:
            # Lấy năm của học kỳ (ví dụ: 'HK1 2023-2024' -> 2023)
            # Giả sử hàm parse_semester_code trả về (year, sem)
            df['current_year'] = df['HOC_KY'].apply(lambda x: int(x.split()[1].split('-')[0]) if isinstance(x, str) else 2020)
            
            # Sinh viên năm thứ mấy
            df['student_year_num'] = df['current_year'] - df['NAM_TUYENSINH'] + 1
            
            # Nếu năm thứ > 4 (hoặc 5 tùy ngành) -> Nguy cơ cao ("Sinh viên già")
            df['is_overdue_student'] = (df['student_year_num'] > 4).astype(int)

        # Clean up các cột tạm
        drop_cols = ['prev_tc_dangky', 'prev_tc_hoanthanh', 'tc_reg_delta', 'current_year', 'student_year_num']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
            
        return df
    
    def _create_advanced_features(self, df):
        """
        Tạo các biến nâng cao: Z-Score, Trend Slope, Volatility
        """
        # Đảm bảo sort đúng để dùng rolling window
        if 'MA_SO_SV' in df.columns and 'HOC_KY' in df.columns:
            # Lưu ý: Sort ở đây chỉ tạm thời cho tính toán, cẩn thận không làm sai lệch index gốc nếu cần
            df = df.sort_values(['MA_SO_SV', 'HOC_KY'])

        # 1. GPA Z-Score (So sánh với mặt bằng chung của kỳ học đó)
        if 'GPA' in df.columns and 'HOC_KY' in df.columns:
            # Tính Mean và Std của GPA theo từng kỳ (HOC_KY)
            # transform giúp map giá trị ngược lại kích thước ban đầu của df
            sem_mean = df.groupby('HOC_KY')['GPA'].transform('mean')
            sem_std = df.groupby('HOC_KY')['GPA'].transform('std')
            
            df['gpa_zscore'] = (df['GPA'] - sem_mean) / (sem_std + 1e-5)
            
            # Xếp hạng phần trăm trong kỳ
            df['gpa_percentile'] = df.groupby('HOC_KY')['GPA'].transform(lambda x: x.rank(pct=True))

        # 2. GPA Volatility (Độ ổn định - Std tích lũy)
        if 'GPA' in df.columns:
            df['gpa_volatility'] = df.groupby('MA_SO_SV')['GPA'].expanding().std().reset_index(level=0, drop=True)
            df['gpa_volatility'] = df['gpa_volatility'].fillna(0)

        # 3. GPA Trend Slope (Xu hướng: đang tiến bộ hay thụt lùi)
        # Lưu ý: Tính toán này có thể chậm với dữ liệu lớn
        if 'GPA' in df.columns:
            # Tính trung bình 3 kỳ gần nhất để làm mượt
            df['gpa_last_3_avg'] = df.groupby('MA_SO_SV')['GPA'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            
            # Tính slope trên cửa sổ 4 kỳ
            # Chú ý: Cần import calculate_slope đã định nghĩa ở trên
            # Để tối ưu tốc độ, có thể comment phần này nếu dữ liệu quá lớn > 500k dòng
            df['gpa_trend_slope'] = df.groupby('MA_SO_SV')['GPA'].transform(
                lambda x: x.rolling(window=4, min_periods=2).apply(calculate_slope, raw=False)
            )
            df['gpa_trend_slope'] = df['gpa_trend_slope'].fillna(0)

        return df
    
    def _create_aggregated_features(self, df):
        if 'MA_SO_SV' in df.columns:
            if 'semester_order' in df.columns:
                df = df.sort_values(['MA_SO_SV', 'semester_order'])
            else:
                df = df.sort_values(['MA_SO_SV'])

            
            if 'TC_HOANTHANH' in df.columns:
                df['total_tc_completed'] = df.groupby('MA_SO_SV')['TC_HOANTHANH'].cumsum()
                df['total_tc_completed_lag1'] = df.groupby('MA_SO_SV')['total_tc_completed'].shift(1)
                df['total_tc_completed_lag1'] = df['total_tc_completed_lag1'].fillna(0)
            
            if 'completion_rate' in df.columns:
                df['avg_completion_rate'] = df.groupby('MA_SO_SV')['completion_rate'].expanding().mean().reset_index(0, drop=True)
                df['avg_completion_rate_lag1'] = df.groupby('MA_SO_SV')['avg_completion_rate'].shift(1)
                df['avg_completion_rate_lag1'] = df['avg_completion_rate_lag1'].bfill().fillna(0)
            
            if 'GPA' in df.columns:
                df['avg_gpa'] = df.groupby('MA_SO_SV')['GPA'].expanding().mean().reset_index(0, drop=True)
                df['avg_gpa_lag1'] = df.groupby('MA_SO_SV')['avg_gpa'].shift(1)
                df['avg_gpa_lag1'] = df['avg_gpa_lag1'].bfill().fillna(0)

            
            df['num_previous_semesters'] = df.groupby('MA_SO_SV').cumcount()
            
            if 'GPA' in df.columns:
                df['gpa_trend'] = df.groupby('MA_SO_SV')['GPA'].diff()
                df['gpa_trend'] = df['gpa_trend'].fillna(0)
                
            if 'completion_rate' in df.columns:
                df['completion_rate_trend'] = df.groupby('MA_SO_SV')['completion_rate'].diff()
                df['completion_rate_trend'] = df['completion_rate_trend'].fillna(0)
        
        return df
    
    def create_test_features(self, test_df, train_df):
        train_df = self.create_features(train_df, is_training=True)
        train_df = train_df.sort_values(['MA_SO_SV', 'semester_order'])
        last_records = train_df.groupby('MA_SO_SV').last().reset_index()
        
        admission_info = train_df[['MA_SO_SV', 'NAM_TUYENSINH', 'PTXT', 'TOHOP_XT', 
                                   'DIEM_TRUNGTUYEN', 'DIEM_CHUAN']].drop_duplicates('MA_SO_SV')
        
        test_features = test_df.merge(
            admission_info,
            on='MA_SO_SV',
            how='left'
        )
        
        test_features = test_features.merge(
            last_records[['MA_SO_SV', 'CPA', 'GPA',
                         'total_tc_completed_lag1', 'avg_completion_rate_lag1',
                         'avg_gpa_lag1', 'num_previous_semesters']],
            on='MA_SO_SV',
            how='left',
            suffixes=('', '_last')
        )
        
        test_features['CPA'] = test_features['CPA'].fillna(0)
        test_features['GPA'] = test_features['GPA'].fillna(0)
        test_features['total_tc_completed_lag1'] = test_features['total_tc_completed_lag1'].fillna(0)
        test_features['avg_completion_rate_lag1'] = test_features['avg_completion_rate_lag1'].fillna(0)
        test_features['avg_gpa_lag1'] = test_features['avg_gpa_lag1'].fillna(0)
        test_features['num_previous_semesters'] = test_features['num_previous_semesters'].fillna(0)
        
        test_features = self.create_features(test_features, is_training=False)
        
        return test_features
    
    def get_feature_columns(self, df, exclude_cols=None):
        if exclude_cols is None:
            exclude_cols = [
                'MA_SO_SV',
                'HOC_KY',
                'TC_HOANTHANH',
                'semester_order',
                'completion_rate',
                'tc_failed',
                'total_tc_completed',
                'avg_completion_rate',
                'completion_rate_trend'
            ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
        
        return feature_cols, categorical_cols


def prepare_features_for_modeling(train_df, valid_df, target_col='TC_HOANTHANH'):
    engineer = FeatureEngineer()
    train_df = engineer.create_features(train_df, is_training=True)
    valid_df = engineer.create_features(valid_df, is_training=False)
    feature_cols, categorical_cols = engineer.get_feature_columns(train_df)
    
    X_train = train_df[feature_cols]
    X_valid = valid_df[feature_cols]
    y_train = train_df[target_col]
    y_valid = valid_df[target_col]
    
    return X_train, X_valid, y_train, y_valid, feature_cols, categorical_cols