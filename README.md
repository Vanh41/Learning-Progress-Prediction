# Learning Progress Prediction
Dự án dự đoán số tín chỉ hoàn thành của sinh viên dựa trên dữ liệu học tập và tuyển sinh.
## Run
### 1. Cài đặt môi trường
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
# Cài đặt thư viện
pip install -r requirements.txt
```



###  2. Thư mục `notebooks/final` quá trình phân tính và khai phá dữ liệu 

| File | Mô tả |
|------|-------|
| `DataCleaning.ipynb` | Pipeline tiền xử lý dữ liệu từ raw data |
| `Dataflow2026EDA.ipynb` | Pipeline EDA |
| `FeatureEngineering.ipynb` | Pipeline cho phần FeatureEngineering |

### 3. Chạy huấn luyện model
```bash
# Chạy với model mặc định (XGBoost)
python main.py --save_model

# Chạy với model cụ thể
python main.py --model_type lightgbm --save_model
python main.py --model_type catboost --save_model

# Chạy với ensemble
python main.py --ensemble --save_model

# Tối ưu với Optuna
python main.py --optimize --model_type xgboost --n_trials 100 --timeout 7200

# Chạy dashboard
streamlit run app/dashboard.py
```

### 4. Chạy Dashboard
```bash
streamlit run app/dashboard.py
```


## Mô tả các file chính

### Thư mục `src/`

| File | Mô tả |
|------|-------|
| `config.py` | Cấu hình đường dẫn, tham số model, hằng số |
| `data_loader.py` | Load và tiền xử lý dữ liệu từ CSV |
| `features.py` | Tạo features: lag, trend, risk indicators |
| `models.py` | Định nghĩa và huấn luyện models (XGBoost, LightGBM, CatBoost) |
| `evaluation.py` | Tính metrics (RMSE, R², MAE) và vẽ biểu đồ |
| `optimization.py` | Tối ưu hyperparameters với Optuna |
| `utils.py` | Các hàm tiện ích: save/load model, logging |

### Các file khác

| File | Mô tả |
|------|-------|
| `dashboard.py` | Dashboard Streamlit để visualize kết quả |
| `comprehensive_analysis.ipynb` | Notebook phân tích và thử nghiệm |
| `main.py` | Script chạy toàn bộ pipeline |
| `requirements.txt` | Danh sách thư viện cần thiết |

## Pipeline chính

```
Data Loading → Feature Engineering → Model Training → Evaluation → Prediction
     ↓              ↓                      ↓              ↓            ↓
data_loader.py → features.py → models.py/optimization.py → evaluation.py → submission.csv
```

## Features chính

- **Lag features**: GPA, CPA, tín chỉ kỳ trước
- **Trend features**: Độ dốc GPA, độ biến động
- **Risk features**: Tỷ lệ fail tích lũy, recovery signals
- **Admission features**: Điểm tuyển sinh, khoảng cách điểm chuẩn

## Models

- **XGBoost**: Model chính với categorical encoding
- **LightGBM**: Model hỗ trợ
- **CatBoost**: Model ensemble
- Hỗ trợ ensemble weighted predictions

## Metrics đánh giá

- RMSE (Root Mean Squared Error)

## Lưu ý

- Dữ liệu được sắp xếp theo time-series (semester_order)
- Categorical features (PTXT, TOHOP_XT) được encode tự động
- Model hỗ trợ early stopping để tránh overfitting
- Kết quả được clip trong khoảng [0, TC_DANGKY]

## Tùy chỉnh

Chỉnh sửa file `src/config.py` để thay đổi:
- Tham số model (learning_rate, max_depth, ...)
- Số trial cho Optuna
- Đường dẫn dữ liệu
- Random seed
