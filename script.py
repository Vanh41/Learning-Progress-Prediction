import pandas as pd
from src.features import FeatureEngineer  # Import class bạn đã gửi

# 1. Đọc dữ liệu thô
academic_df = pd.read_csv('./data/raw/academic_records.csv')
admission_df = pd.read_csv('./data/raw/admission.csv')

# 2. Hợp nhất (Merge) dữ liệu trước khi đưa vào FeatureEngineer
# Vì FeatureEngineer cần cả thông tin từ admission và academic
df_raw = academic_df.merge(admission_df, on='MA_SO_SV', how='left')

# 3. Khởi tạo bộ tạo đặc trưng
engineer = FeatureEngineer()

# 4. Tạo đặc trưng (Chạy hàm mà bạn muốn giải thích)
df_featured = engineer.create_features(df_raw)

# 5. Xem kết quả
print("--- Danh sách các cột mới đã được tạo ---")
new_cols = [c for c in df_featured.columns if c not in df_raw.columns]
print(new_cols)

print("\n--- Dữ liệu của 5 dòng đầu tiên (Các cột mới) ---")
# Hiển thị các cột mới tạo để kiểm tra logic
print(df_featured[new_cols].head())

# 6. Lưu ra file CSV để soi kỹ hơn bằng Excel
df_featured.to_csv('data_after_feature.csv', index=False)
print("\nĐã lưu file 'data_after_feature.csv'. Bạn có thể mở bằng Excel để xem.")