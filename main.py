import pandas as pd
import numpy as np
from pathlib import Path
import argparse

from src.config import RANDOM_STATE, get_model_output_dir
from src.utils import set_seed, save_model, save_submission
from src.data_loader import load_and_prepare_data
from src.features import FeatureEngineer, prepare_features_for_modeling
from src.models import create_model, create_ensemble
from src.optimization import optimize_model
from src.evaluation import create_evaluation_report, print_metrics
from src.utils import log_experiment
from datetime import datetime



def main(args):
    print("LEARNING PROGRESS PREDICTION - DATAFLOW TEAM")
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
    
    # Step 1: Load and prepare data
    print("\n[STEP 1] Loading and preparing data...")
    train_df, valid_df, test_df = load_and_prepare_data()
    
    # Step 2: Feature engineering
    print("\n[STEP 2] Creating features...")
    X_train, X_valid, y_train, y_valid, feature_cols, categorical_cols = prepare_features_for_modeling(
        train_df, valid_df
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_valid.shape}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Categorical features: {categorical_cols}")
    
    # Step 3: Model training
    print("\n[STEP 3] Training model...")
    
    if args.optimize:
        # Hyperparameter optimization
        print("Running hyperparameter optimization...")
        best_params, best_model, optimizer = optimize_model(
            args.model_type,
            X_train, y_train,
            X_valid, y_valid,
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
    
    # Step 4: Evaluation
    print("\n[STEP 4] Evaluating model...")
    
    # Predictions on validation set
    y_pred_valid = trainer.predict(X_valid, categorical_cols)
    
    # Create evaluation report (save to model-specific directory)
    metrics = create_evaluation_report(
        y_valid, y_pred_valid,
        feature_names=feature_cols if not args.ensemble else None,
        model=trainer.model if not args.ensemble else None,
        save_dir=model_output_dir  # ← Key change: pass model-specific directory
    )
    
    # Save metrics to file
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(model_output_dir / 'metrics.csv', index=False)
    print(f"Metrics saved to: {model_output_dir / 'metrics.csv'}")
    
    # Step 5: Predictions on test set
    if test_df is not None and len(test_df) > 0:
        print("\n[STEP 5] Making predictions on test set...")
        
        # Create features for test set
        engineer = FeatureEngineer()
        test_features = engineer.create_test_features(test_df, train_df)
        
        # Ensure test has all train features
        missing_cols = set(feature_cols) - set(test_features.columns)
        for col in missing_cols:
            test_features[col] = 0

        # Ensure test features have same columns as training
        X_test = test_features[feature_cols]
        
        # Fix NaN in categorical features for CatBoost
        for col in categorical_cols:
            if col in X_test.columns:
                X_test[col] = X_test[col].astype(str).fillna("missing")

        # Make predictions
        y_pred_test = trainer.predict(X_test, categorical_cols)
        
        # Save submission to model-specific directory
        submission = pd.DataFrame({
            'MA_SO_SV': test_df['MA_SO_SV'],
            'PRED_TC_HOANTHANH': y_pred_test.astype(int)
        })
        submission_path = model_output_dir / f'{args.team_name}_submission.csv'
        submission.to_csv(submission_path, index=False)
        
        print(f"Submission saved to: {submission_path}")
    
    # Step 6: Save model
    if args.save_model:
        print("\n[STEP 6] Saving model...")
        model_filename = f"{model_name}_model.pkl"
        model_path = model_output_dir / model_filename
        
        # Use custom save since we want to save to model-specific directory
        import joblib
        joblib.dump(trainer, model_path)
        print(f"Model saved to: {model_path}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"All outputs saved to: {model_output_dir}")
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
        "output_dir": str(model_output_dir)
    }
    log_file = log_experiment(experiment_name, metrics, params)
    print(f"Experiment logged at: {log_file}")
    
    return trainer, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning Progress Prediction Pipeline')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='xgboost',
                       choices=['xgboost', 'lightgbm', 'catboost', 'random_forest'],
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