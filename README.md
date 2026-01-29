## 📂 Cấu trúc Dự án (Project Structure)

Dự án được tổ chức theo cấu trúc tiêu chuẩn của Data Science để đảm bảo tính tái lập (reproducibility) và dễ dàng mở rộng.

```text
Learning-process-prediction/
├── data/
│   ├── raw/                   # 🔒 Dữ liệu thô (Immutable) - KHÔNG ĐƯỢC SỬA FILE Ở ĐÂY
│   │   ├── admission.csv
│   │   └── academic_records.csv
│   ├── external/              # 🌍 Dữ liệu bên ngoài (Thời tiết, kinh tế, điểm chuẩn...)
│   └── processed/             # ⚙️ Dữ liệu đã làm sạch & Feature Engineering (Dùng để train)
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── notebooks/                 # 📓 Jupyter Notebooks
│   ├── experimental/          # Khu vực nháp (Sandbox) - Đặt tên: [TenTv]_[TenTask].ipynb
│   └── final/                 # Notebook sạch để báo cáo/thuyết trình (Đã clear output)
│       ├── 1_EDA_Story.ipynb         # Phân tích khám phá & Câu chuyện dữ liệu
│       ├── 2_Modeling_Process.ipynb  # Quá trình huấn luyện & So sánh model
│       └── 3_Policy_Analysis.ipynb   # Đề xuất giải pháp & Phân tích tác động
│
├── src/                       # 🧠 MÃ NGUỒN CHÍNH (Pipeline)
│   ├── __init__.py
│   ├── config.py              # Cấu hình toàn cục (Path, Random Seed, Constants)
│   ├── data_loader.py         # Pipeline: Đọc CSV -> Clean -> Merge -> Split
│   ├── features.py            # Feature Engineering: Tạo Lag, Trend, Ratio features
│   ├── models.py              # Model Architecture: Định nghĩa XGBoost, LSTM, etc.
│   ├── optimization.py        # Tuning: Chạy Optuna/GridSearch tối ưu tham số
│   ├── evaluation.py          # Metrics: Tính RMSE, R2, SHAP, LIME
│   └── utils.py               # Tiện ích: Logger, Save/Load Model, Helper functions
│
├── app/                       # 📊 Dashboard Application
│   └── dashboard.py           # Mã nguồn ứng dụng Streamlit demo kết quả
│
├── models/                    # 💾 Nơi lưu trữ Model đã huấn luyện (.pkl, .json, .h5)
├── output/                    # 📤 Kết quả đầu ra
│   ├── submission.csv         # File nộp bài cuối cùng
│   └── figures/               # Biểu đồ xuất ra từ code (để chèn vào báo cáo)
│
├── main.py                    # 🚀 ENTRY POINT: Script chạy toàn bộ quy trình từ A-Z
├── requirements.txt           # Danh sách các thư viện cần thiết
└── README.md                  # Tài liệu hướng dẫn sử dụng dự án
```

# 🛠 Hướng dẫn Cài đặt Môi trường (Setup Environment)

Dự án sử dụng thư viện `virtualenv` để quản lý gói cài đặt. Vui lòng làm theo các bước sau trước khi code.

### Bước 1: Cài đặt công cụ virtualenv
Nếu máy bạn chưa có thư viện này, hãy cài đặt nó (chỉ cần làm 1 lần):
```bash
pip install virtualenv
```

### Bước 2: Tạo môi trường ảo
Tại thư mục gốc của dự án (`Learning-process-prediction/`), chạy lệnh:
```bash
# Tạo thư mục môi trường tên là 'venv'
virtualenv venv
```

### Bước 3: Kích hoạt môi trường (Activate)
*Mỗi lần bắt đầu làm việc, bạn phải chạy lệnh này.*

*   **Đối với Windows (Command Prompt/PowerShell):**
    ```bash
    .\venv\Scripts\activate
    ```
    *(Nếu thấy dấu `(venv)` hiện ở đầu dòng lệnh là thành công)*

*   **Đối với macOS / Linux:**
    ```bash
    source venv/bin/activate
    ```

### Bước 4: Cài đặt thư viện dự án
Sau khi kích hoạt môi trường, hãy cài đặt các thư viện cần thiết từ file `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Bước 5: Thêm Kernel vào Jupyter Notebook (QUAN TRỌNG)
Để chạy được Notebooks trong thư mục `notebooks/` với môi trường ảo vừa tạo:

1.  Cài đặt ipykernel:
    ```bash
    pip install ipykernel
    ```
2.  Gắn môi trường vào Jupyter:
    ```bash
    python -m ipykernel install --user --name=venv_learning_prediction --display-name "Python (Learning Prediction)"
    ```
3.  Khi mở Jupyter Notebook, chọn Kernel: **Kernel** -> **Change kernel** -> **Python (Learning Prediction)**.

---
### 🛑 Cách thoát môi trường
Khi làm xong việc, chạy lệnh:
```bash
deactivate
```

---

### 💡 Lưu ý cho Leader (Role A):

1.  **File `.gitignore`**: Hãy chắc chắn file `.gitignore` của bạn đã có dòng `venv/` (như mình đã đưa ở câu trả lời trước) để không lỡ tay push cả thư viện lên Github.
2.  **Cập nhật `requirements.txt`**: Vì team làm việc song song, thỉnh thoảng sẽ có người cài thêm thư viện mới (ví dụ `matplotlib`, `seaborn`). Hãy nhắc team chạy lệnh sau trước khi Push code để cập nhật danh sách thư viện cho người khác:
    ```bash
    pip freeze > requirements.txt
<<<<<<< HEAD
    ```
=======
    ```


>>>>>>> feature/overview
