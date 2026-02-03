# Hướng dẫn chạy

# Chạy với model mặc định (XGBoost)
python main.py --save_model
hoặc
python main.py --team_name Multour

# Chạy với model cụ thể
python main.py --model_type lightgbm --save_model
python main.py --model_type catboost --save_model

# Chạy với ensemble
python main.py --ensemble --save_model

# Tối ưu với Optuna
python main.py --optimize --model_type xgboost --n_trials 100 --timeout 7200

# Chạy dashboard
streamlit run app/dashboard.py


