import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.features import FeatureEngineer
from src.models import create_ensemble

# 1. CHUẨN BỊ DỮ LIỆU
print("--- Đang tải dữ liệu ---")
train_academic = pd.read_csv('./data/raw/academic_records.csv')
admission = pd.read_csv('./data/raw/admission.csv')
test_df = pd.read_csv('./data/raw/test.csv')

# Merge dữ liệu để có đầy đủ thông tin
train_raw = train_academic.merge(admission, on='MA_SO_SV', how='left')
train_raw['semester_order'] = train_raw.groupby('MA_SO_SV').cumcount()

# 2. FEATURE ENGINEERING
engineer = FeatureEngineer()
# Giả sử chúng ta dùng chính train_raw để minh họa quy trình
train_featured = engineer.create_features(train_raw)
feature_cols = engineer.get_feature_columns(train_featured)
categorical_cols = []

# 3. HUẤN LUYỆN MÔ HÌNH (Dùng Ensemble để kiểm tra tổng thể)
print("\n--- Đang huấn luyện mô hình Ensemble ---")
X = train_featured[feature_cols]
y = train_featured['TC_HOANTHANH']

ensemble = create_ensemble()
ensemble.train(X, y, categorical_cols=categorical_cols)

# 4. DỰ BÁO TRÊN TẬP TEST
# Lưu ý: create_test_features cần cả tập test và train để lấy lịch sử SV
test_featured = engineer.create_test_features(test_df, train_featured)
missing_cols = set(feature_cols) - set(test_featured.columns)
for col in missing_cols:
    test_featured[col] = 0
X_test = test_featured[feature_cols]

# FIX CatBoost categorical NaN / float
for col in categorical_cols:
    X_test[col] = X_test[col].fillna('UNKNOWN').astype(str)

y_pred = ensemble.predict(X_test, categorical_cols=categorical_cols)


# 5. KIỂM TRA TÍNH HỢP LÝ (SANITY CHECK)
print("\n--- Đang kiểm tra tính hợp lý của dự báo ---")
results = test_df.copy()
results['Dự_báo_TC_Hoàn_Thành'] = y_pred

# Kiểm tra lỗi Logic: Hoàn thành > Đăng ký
logic_error = results[results['Dự_báo_TC_Hoàn_Thành'] > results['TC_DANGKY']]

print(f"Tổng số dòng dự báo: {len(results)}")
if len(logic_error) > 0:
    print(f"⚠️ CẢNH BÁO: Có {len(logic_error)} dòng dự báo vô lý (Hoàn thành > Đăng ký)!")
    print(logic_error[['MA_SO_SV', 'TC_DANGKY', 'Dự_báo_TC_Hoàn_Thành']].head())
else:
    print("✅ Chúc mừng: Tất cả dự báo đều nằm trong phạm vi số tín chỉ đăng ký.")

# 6. KIỂM TRA ĐỘ QUAN TRỌNG CỦA ĐẶC TRƯNG (Dùng model đầu tiên trong ensemble - thường là XGBoost)
print("\n--- Đang phân tích độ quan trọng của đặc trưng ---")
first_model = ensemble.models[0]
importance = first_model.get_feature_importance()

# Sắp xếp và vẽ biểu đồ
feat_importance = pd.DataFrame({'Feature': feature_cols, 'Importance': importance})
feat_importance = feat_importance.sort_values(by='Importance', ascending=False).head(10)



plt.figure(figsize=(10, 6))
plt.barh(feat_importance['Feature'], feat_importance['Importance'], color='skyblue')
plt.xlabel('Độ quan trọng')
plt.title('Top 10 đặc trưng ảnh hưởng nhất đến kết quả dự báo')
plt.gca().invert_yaxis()
plt.show()

# 7. KIỂM TRA MÃ HÓA (LABEL ENCODING)
print("\n--- Kiểm tra mã hóa danh mục ---")
for col, encoder in first_model.label_encoders.items():
    print(f"Cột [{col}]: Mã hóa được {len(encoder.classes_)} nhãn.")