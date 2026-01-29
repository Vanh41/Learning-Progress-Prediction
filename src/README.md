# Hướng dẫn chạy

# Chạy với model mặc định (XGBoost)
python main.py --save_model
hoặc
python main.py --team_name Multour

# Chạy với model cụ thể
python main.py --model_type lightgbm --save_model
python main.py --model_type catboost --save_model
python main.py --model_type random_forest --save_model

# Chạy với ensemble
python main.py --ensemble --save_model

# Tối ưu với Optuna
python main.py --optimize --model_type xgboost --n_trials 100 --timeout 7200

# Chạy dashboard
streamlit run app/dashboard.py


main.py: gọi tất cả các module khác để chạy pipeline hoàn chỉnh

config: 
    TRAIN_START_YEAR = 2020  Năm bắt đầu lấy dữ liệu train (2020)
    TRAIN_END_SEMESTER = 'HK1 2023-2024'  Học kỳ cuối cùng của train set
    Tất cả data từ 2020 đến HK1 2023-2024 -> Train
    VALID_SEMESTER = 'HK2 2023-2024'     Học kỳ validation (HK2 2023-2024)  
    TEST_SEMESTER = 'HK1 2024-2025'   Học kỳ test (HK1 2024-2025)
    N_SPLITS = 5    Số folds cho cross-validation (nếu dùng)
    max_depth: Độ sâu tối đa của cây  == depth
    learning_rate: Tốc độ học
    n_estimators: Số lượng trees  == iterations
    verbose=False: Không in log khi training
    OPTUNA_N_TRIALS: Số lần thử hyperparameter combinations
    OPTUNA_TIMEOUT: Thời gian tối đa (giây)


utils:
    parse_semester_code: HK1 2020-2021 -> 2020,1
    get_semester_order: 20201
    calculate_semester_from_admission: Tính xem sinh viên đang ở học kỳ thứ mấy kể từ khi nhập học
    log_experiment: Ghi lại lịch sử tất cả experiments 
    memory_usage: Tính memory usage của DataFrame

data_loader: Module xử lý loading, cleaning, merging và splitting data (Xóa duplicates, Fill missing values, Data validation)
    Join academic records với admission data
    Temporal split (dựa vào thời gian)
    One-liner để gọi toàn bộ pipeline data loading
    Raw CSVs -> load_raw_data() -> clean_data() -> merge_data() -> split_data() -> train_df, valid_df, test_df

evaluation: đánh giá và visualize model performance
    đánh giá model với metrics và visualizations
    plot_predictions: Biểu đồ so sánh giá trị dự đoán vs thực tế
    plot_residuals: Kiểm tra phân phối sai số
    plot_feature_importance: Top N features quan trọng nhất
    plot_error_distribution_by_groups: So sánh lỗi dự đoán giữa các nhóm
    create_evaluation_report: đánh giá đầy đủ với 3-4 biểu đồ

features: 
    _create_admission_features: 
                                diem_vuot_chuan: Điểm vượt điểm chuẩn; 
                                diem_ratio: tỷ lệ điểm trúng tuyển / điểm chuẩn;
                                nam_tuoi: số năm kể từ khi nhập học (xác định sinh viên năm 4, 5 thì số TC đăng kí ít hơn);
    _create_academic_features: 
                               gpa_cpa_diff: GPA tăng/giảm so với CPA
                               gpa_cpa_ratio: tỷ lệ GPA / CPA
                               tc_dangky_high, tc_dangky_low: Binary flags cho số TC đăng ký cao/thấp (đăng kí nhiều: 0, ít: 1)
                               completion_rate: Tỷ lệ hoàn thành (tchoanthanh / tcdangky)
                               tc_failed: Số TC trượt (tcdangky - tc_hoanthanh)
    _create_temporal_features: 
                               semester_number: kì học hiện tại của sinh viên (kể từ khi nhập học);
                               hoc_ky_nam, hoc_ky_so: Parse năm và số học kỳ (HK1 2020-2021 -> 2020 1);
                               is_semester_2: binary flag cho học kỳ 2 (hoc_ky_so: 1|2 -> is_semester_2: 0|1) -> học pattern hk1 và hk2 khác nhau;
    _create_performance_features: phân loại học lực thành categories
                                8 features (4 cho CPA, 4 cho GPA) (>=3.6, >=3.2, >=2.5, <2.5) - binary (CPA=2.0 nhưng GPA=3.5 → Sinh viên đang tiến bộ);
    _create_aggregated_features: 
                                sort data: sort by student → sort by semester: aggregated features cần data theo thứ tự thời gian;
                                total_tc_completed: tổng tín chỉ đã hoàn thành đến thời điểm hiện tại;
                                total_tc_completed_lag1: dịch xuống 1 row (lấy giá trị học kỳ trước) (học kì đầu không có lag -> fill 0);
                                avg_completion_rate: Tỷ lệ hoàn thành trung bình trong toàn bộ quá trình học;
                                avg_completion_rate_lag1: dịch xuống 1 row (fill NaN với value kế tiếp: NaN 3 4 -> 3 3 4);
                                avg_gpa:  giống avg_completion_rate;
                                num_previous_semesters: kinh nghiệm học tập (càng nhiều semester trước -> càng có kinh nghiệm)
                                gpa_trend: >0 GPA tăng, <0 GPA giảm;
                                completion_rate_trend: xu hướng của tỉ lệ hoàn thành;
                                create_test_features: tạo features cho test set (để predict): test set chỉ có (MA_SO_SV, HOC_KY, TC_DANGKY) -> lấy features từ train set học kỳ cuối cùng của mỗi SV -> merge với test (sinh viên mới fill 0);
    get_feature_columns: Xác định columns nào là features, columns nào là categorical 
                                feature_cols: lấy tất cả columns trừ exclude_cols;
                                categorical_cols: Chọn columns có dtype là object (PTXT, TOHOP_XT, ...) -> encode thành số;
                                prepare_features_for_modeling: One-liner để prepare features cho modeling;

model: 
    _encode_categorical_features: string -> integer, testing mode: test set có thể chứa categories chưa thấy trong train;
    train: Train model với data đã prepare;
    predict: dự đoán;
    get_feature_importance: lấy feature importance từ model (mức độ đóng góp của feature vào predictions, dựa trên số lần feature được dùng để split, gain khi split);
    EnsembleModel: Kết hợp predictions từ nhiều models, get_individual_predictions: Xem prediction của từng model riêng lẻ

optimization:
    



















