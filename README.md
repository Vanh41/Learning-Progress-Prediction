# MulTour - Learning Progress Prediction
Dự án dự đoán số tín chỉ hoàn thành của sinh viên dựa trên dữ liệu học tập và tuyển sinh.
## Run


Sau khi clone repository, bạn có thể cài đặt các dependencies cục bộ trên Python>=3.11 như sau:



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



###  2. Thư mục `notebooks/final` quá trình phân tính và khai phá dữ liệu, kết quả chạy của các model.

| File | Mô tả |
|------|-------|
| `DataCleaning.ipynb` | Pipeline tiền xử lý dữ liệu từ raw data |
| `Dataflow2026EDA.ipynb` | Pipeline EDA |
| `FeatureEngineering.ipynb` | Pipeline cho phần FeatureEngineering |
| `final_lightgbm.ipynb` | Pipeline training cho model LightGBM |
| `final_xgboost.ipynb` | Pipeline training cho model XGBoost |
| `final_catboost.ipynb` | Pipeline training cho model CatBoost |



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
```text
DATAFLOW_TEAM_NAME/
├── data/
│   ├── raw/                   # Chứa file gốc: admission.csv, academic_records.csv
│   └── processed/             # Dữ liệu đã sạch (để train model)
│
├── notebooks/                 # Nơi chạy thử nghiệm (Jupyter Notebooks)
│   ├── experimental/          # Nháp 
│   └── final/                 # Notebook sạch sẽ dùng để nộp/thuyết trình

├── src/                       # MÃ NGUỒN CHÍNH (Các hàm tái sử dụng)
│   ├── __init__.py
│   ├── config.py              # Cấu hình đường dẫn, tham số global
│   ├── data_loader.py         # Hàm đọc, làm sạch và merge dữ liệu
│   ├── features.py            # Hàm tạo biến đặc trưng (Feature Engineering)
│   ├── models.py              # Hàm định nghĩa model, train và predict
│   ├── optimization.py        # Hàm chạy Optuna tối ưu tham số
│   ├── evaluation.py          # Hàm tính metric (RMSE, R2) và vẽ biểu đồ lỗi
│   └── utils.py               # Các hàm phụ trợ (Lưu file, set seed...)
│
├── app/                       # Dashboard (Streamlit)
│   └── dashboard.py           
│
├── models/                    # Nơi lưu file model đã train (.pkl, .json)
├── output/                    # Kết quả output (submission.csv, charts)
├── main.py                    # FILE CHẠY CHÍNH (Pipeline)
├── main_from_processed.py     # file mục đích để test pipeline có chạy ổn không
├── requirements.txt           # Danh sách thư viện cần cài
└── README.md                  # Hướng dẫn chạy code
```
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
