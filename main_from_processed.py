import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.metrics import mean_squared_error

from src.config import RANDOM_STATE, get_model_output_dir
from src.utils import set_seed, save_model, save_submission, log_experiment
from src.data_loader import load_and_prepare_data
from src.features import FeatureEngineer, prepare_features_for_modeling
from src.models import create_model, create_ensemble
from src.optimization import optimize_model
from src.evaluation import create_evaluation_report, print_metrics, calculate_metrics
from datetime import datetime


def main(args):
    print("="*80)
    print("LEARNING PROGRESS PREDICTION - DATAFLOW 2026")
    print("="*80)
    set_seed(RANDOM_STATE)
    
    # Determine model name for output directory
    if args.ensemble:
        model_name = 'ensemble'
    else:
        model_name = args.model_type
    
    if args.optimize:
        model_name += '_optimized'
    
    # Create output directory for this model
    model_output_dir = get_model_output_dir(model_name)
    print(f"\nModel output directory: {model_output_dir}")
    
    # STEP 1 & 2: Load processed data
    print("\n" + "="*80)
    print("[STEP 1 & 2] Loading processed data...")
    print("="*80)
    
    # Load trực tiếp dữ liệu đã xử lý
    train_feat = pd.read_csv('./data/processed/train_final.csv')
    test_feat = pd.read_csv('./data/processedtest_final.csv')
    
    # Giả sử bạn dùng 1 phần train làm validation hoặc đã có valid_final.csv
    # Ở đây tôi ví dụ tách validation từ train nếu bạn chưa có file valid riêng
    from sklearn.model_selection import train_test_split
    train_feat, valid_feat = train_test_split(train_feat, test_size=0.2, random_state=RANDOM_STATE)

    # Khởi tạo Engineer để lấy danh sách cột hợp lệ
    engineer = FeatureEngineer()
    feature_cols = engineer.get_feature_columns(train_feat)
    categorical_cols = [c for c in feature_cols if c in engineer.cat_cols]
    
    target_col = 'COMPLETION_RATE' # Hoặc 'TC_HOANTHANH' tùy mục tiêu của bạn

    # Prepare X, y
    X_train = train_feat[feature_cols].copy()
    y_train = train_feat[target_col]

    X_valid = valid_feat[feature_cols].copy()
    y_valid = valid_feat[target_col]

    X_test = test_feat[feature_cols].copy()

    # Encode categorical columns nếu cần
    from sklearn.preprocessing import OrdinalEncoder
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if categorical_cols:
        X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols].astype(str))
        X_valid[categorical_cols] = encoder.transform(X_valid[categorical_cols].astype(str))
        X_test[categorical_cols] = encoder.transform(X_test[categorical_cols].astype(str))

    # Gán lại các biến cần thiết cho các bước sau
    valid_full_df = valid_feat
    test_full_df = test_feat
    
    # STEP 3: Model training
    print("\n" + "="*80)
    print("[STEP 3] Training model...")
    print("="*80)
    
    if args.optimize:
        # Hyperparameter optimization
        print("Running hyperparameter optimization...")
        
        # Cần valid_credits_dangky cho optimization
        valid_credits_dangky = valid_full_df['TC_DANGKY'].values
        
        best_params, best_model, optimizer = optimize_model(
            args.model_type,
            X_train, y_train,
            X_valid, y_valid,
            valid_credits_dangky,
            categorical_cols,
            n_trials=args.n_trials,
            timeout=args.timeout
        )
        
        # Save optimization plots
        try:
            optimizer.plot_optimization_history(save_path=model_output_dir / 'optimization_history.png')
            optimizer.plot_param_importances(save_path=model_output_dir / 'param_importances.png')
        except Exception as e:
            print(f"Warning: Could not save optimization plots: {e}")
        
        # Train final model with best parameters
        trainer = best_model
        trainer.train(X_train, y_train, X_valid, y_valid, categorical_cols)
        
    elif args.ensemble:
        # Ensemble model
        print("Training ensemble model...")
        trainer = create_ensemble()
        trainer.train(X_train, y_train, X_valid, y_valid, categorical_cols)
        
    else:
        # Single model
        print(f"Training {args.model_type} model...")
        trainer = create_model(args.model_type)
        trainer.train(X_train, y_train, X_valid, y_valid, categorical_cols)
    
    # STEP 4: Evaluation
    print("\n" + "="*80)
    print("[STEP 4] Evaluating model...")
    print("="*80)
    
    # Predictions on validation set (predict COMPLETION_RATE)
    y_pred_rate_valid = trainer.predict(X_valid, categorical_cols)
    
    # HẬU XỬ LÝ (Post-Processing Strategy)
    print("\nApplying post-processing to validation predictions...")
    
    # 1. Clip Rate về [0, 1]
    y_pred_rate_valid = np.clip(y_pred_rate_valid, 0, 1)
    
    # 2. Convert về Số tín chỉ: Rate * TC_DANGKY
    valid_credits_dangky = valid_full_df['TC_DANGKY'].values
    y_pred_credits_valid = y_pred_rate_valid * valid_credits_dangky
    
    # 3. Hard Limit: Không vượt quá đăng ký
    y_pred_credits_valid = np.minimum(y_pred_credits_valid, valid_credits_dangky)
    
    # 4. Tính metrics trên CREDITS (không phải rate)
    y_valid_credits = valid_full_df['TC_HOANTHANH'].values
    
    # Calculate metrics
    metrics = calculate_metrics(y_valid_credits, y_pred_credits_valid)
    print_metrics(metrics, title="Validation Performance (Credits)")
    
    # Additional statistics
    mae = np.mean(np.abs(y_valid_credits - y_pred_credits_valid))
    accurate = np.sum(np.abs(y_valid_credits - y_pred_credits_valid) < 1)
    accuracy_pct = accurate / len(y_valid_credits) * 100
    
    print(f"\nAdditional Statistics:")
    print(f"  MAE: {mae:.4f}")
    print(f"  Accuracy (±1 credit): {accuracy_pct:.2f}%")
    
    # Create evaluation report (save to model-specific directory)
    # Pass credits for visualization
    try:
        create_evaluation_report(
            y_valid_credits, 
            y_pred_credits_valid,
            feature_names=feature_cols if not args.ensemble else None,
            model=trainer.model if hasattr(trainer, 'model') and not args.ensemble else None,
            save_dir=model_output_dir
        )
    except Exception as e:
        print(f"Warning: Could not create full evaluation report: {e}")
    
    # Save metrics to file
    metrics_df = pd.DataFrame([metrics])
    metrics_df['MAE'] = mae
    metrics_df['Accuracy_1Credit'] = accuracy_pct
    metrics_df.to_csv(model_output_dir / 'metrics.csv', index=False)
    print(f"\nMetrics saved to: {model_output_dir / 'metrics.csv'}")
    
    # STEP 5: Predictions on test set
    if X_test is not None and test_full_df is not None:
        print("\n" + "="*80)
        print("[STEP 5] Making predictions on test set...")
        print("="*80)
        
        # Make predictions (predict COMPLETION_RATE)
        y_pred_rate_test = trainer.predict(X_test, categorical_cols)
        
        # HẬU XỬ LÝ cho test predictions
        print("\nApplying post-processing to test predictions...")
        
        # 1. Clip Rate về [0, 1]
        y_pred_rate_test = np.clip(y_pred_rate_test, 0, 1)
        
        # 2. Convert về Số tín chỉ
        test_credits_dangky = test_full_df['TC_DANGKY'].values
        y_pred_credits_test = y_pred_rate_test * test_credits_dangky
        
        # 3. Hard Limit: Không vượt quá đăng ký
        y_pred_credits_test = np.minimum(y_pred_credits_test, test_credits_dangky)
        
        print(f"\nTest predictions statistics:")
        print(f"  Mean: {y_pred_credits_test.mean():.2f}")
        print(f"  Std:  {y_pred_credits_test.std():.2f}")
        print(f"  Min:  {y_pred_credits_test.min():.2f}")
        print(f"  Max:  {y_pred_credits_test.max():.2f}")
        
        # Distribution analysis
        print(f"\nPrediction distribution:")
        print(f"  < 5 credits:   {np.sum(y_pred_credits_test < 5)} ({np.sum(y_pred_credits_test < 5) / len(y_pred_credits_test) * 100:.1f}%)")
        print(f"  5-10 credits:  {np.sum((y_pred_credits_test >= 5) & (y_pred_credits_test < 10))} ({np.sum((y_pred_credits_test >= 5) & (y_pred_credits_test < 10)) / len(y_pred_credits_test) * 100:.1f}%)")
        print(f"  10-15 credits: {np.sum((y_pred_credits_test >= 10) & (y_pred_credits_test < 15))} ({np.sum((y_pred_credits_test >= 10) & (y_pred_credits_test < 15)) / len(y_pred_credits_test) * 100:.1f}%)")
        print(f"  >= 15 credits: {np.sum(y_pred_credits_test >= 15)} ({np.sum(y_pred_credits_test >= 15) / len(y_pred_credits_test) * 100:.1f}%)")
        
        # Save submission to model-specific directory
        submission = pd.DataFrame({
            'MA_SO_SV': test_full_df['MA_SO_SV'],
            'PRED_TC_HOANTHANH': y_pred_credits_test
        })
        
        submission_path = model_output_dir / f'{args.team_name}_submission.csv'
        submission.to_csv(submission_path, index=False)
        
        print(f"\nSubmission saved to: {submission_path}")
        print(f"\nFirst 10 predictions:")
        print(submission.head(10))
        print(f"\nLast 10 predictions:")
        print(submission.tail(10))
    else:
        print("\n" + "="*80)
        print("[STEP 5] No test data available - skipping predictions")
        print("="*80)
    
    # STEP 6: Save model
    if args.save_model:
        print("\n" + "="*80)
        print("[STEP 6] Saving model...")
        print("="*80)
        model_filename = f"{model_name}_model.pkl"
        model_path = model_output_dir / model_filename
        
        # Use joblib to save
        import joblib
        joblib.dump(trainer, model_path)
        print(f"Model saved to: {model_path}")
        
        # Also save feature columns and categorical columns
        feature_info = {
            'feature_cols': feature_cols,
            'categorical_cols': categorical_cols
        }
        joblib.dump(feature_info, model_output_dir / 'feature_info.pkl')
        print(f"Feature info saved to: {model_output_dir / 'feature_info.pkl'}")
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nModel Performance Summary:")
    print(f"  Validation RMSE: {metrics['RMSE']:.4f}")
    print(f"  Validation MAE:  {mae:.4f}")
    print(f"  Accuracy (±1):   {accuracy_pct:.2f}%")
    print(f"\nAll outputs saved to: {model_output_dir}")
    print("="*80)

    # Log experiment
    experiment_name = model_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    params = {
        "model_type": args.model_type,
        "ensemble": args.ensemble,
        "optimize": args.optimize,
        "n_trials": args.n_trials if args.optimize else None,
        "timeout": args.timeout if args.optimize else None,
        "random_state": RANDOM_STATE,
        "num_features": len(feature_cols),
        "num_categorical": len(categorical_cols),
        "output_dir": str(model_output_dir)
    }
    
    log_metrics = {
        'RMSE': metrics['RMSE'],
        'MAE': mae,
        'Accuracy_1Credit': accuracy_pct
    }
    
    log_file = log_experiment(experiment_name, log_metrics, params)
    print(f"\nExperiment logged at: {log_file}")
    
    return trainer, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Learning Progress Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train default XGBoost model
  python main.py --save_model
  
  # Train specific model
  python main.py --model_type lightgbm --save_model
  
  # Train ensemble
  python main.py --ensemble --save_model
  
  # Optimize hyperparameters
  python main.py --optimize --model_type xgboost --n_trials 100 --save_model
  
  # Quick test run without saving
  python main.py
        """
    )
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='xgboost',
                       choices=['xgboost', 'lightgbm', 'catboost'],
                       help='Type of model to train')
    parser.add_argument('--ensemble', action='store_true',
                       help='Use ensemble of models')
    
    # Optimization arguments
    parser.add_argument('--optimize', action='store_true',
                       help='Run hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, default=3600,
                       help='Optimization timeout in seconds')
    
    # Output arguments
    parser.add_argument('--team_name', type=str, default='dataflow',
                       help='Team name for submission file')
    parser.add_argument('--save_model', action='store_true',
                       help='Save trained model')
    
    args = parser.parse_args()
    
    # Run pipeline
    trainer, metrics = main(args)