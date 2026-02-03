import optuna
from optuna.samplers import TPESampler
import numpy as np
from sklearn.metrics import mean_squared_error
from src.models import ModelTrainer


class OptunaOptimizer:    
    def __init__(self, model_type='xgboost', n_trials=50, timeout=3600):
        self.model_type = model_type
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params = None
        self.study = None
        
    def _objective(self, trial, X_train, y_train, X_valid, y_valid, 
                   valid_credits_dangky, categorical_cols):
        if self.model_type == 'xgboost':
            params = {
                'n_estimators': 2000,
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                
                # Fixed params
                'objective': 'reg:squarederror',
                'n_jobs': -1,
                'random_state': 42,
                'enable_categorical': True,
                'tree_method': 'hist',
            }
            
        elif self.model_type == 'lightgbm':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'verbose': -1
            }
            
        elif self.model_type == 'catboost':
            params = {
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'iterations': trial.suggest_int('iterations', 100, 500),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_state': 42,
                'verbose': False
            }
            
        elif self.model_type == 'random_forest':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'random_state': 42
            }
        else:
            raise ValueError(f"Optimization not implemented for {self.model_type}")
        
        # Train model with these params
        trainer = ModelTrainer(self.model_type, params)
        trainer.train(X_train, y_train, X_valid, y_valid, categorical_cols)
        
        # Predict on validation set
        y_pred_rate = trainer.predict(X_valid, categorical_cols)
        
        # Convert to credits
        y_pred_rate = np.clip(y_pred_rate, 0, 1)
        y_pred_credits = y_pred_rate * valid_credits_dangky
        y_pred_credits = np.minimum(y_pred_credits, valid_credits_dangky)
        y_valid_credits = y_valid * valid_credits_dangky
        rmse = np.sqrt(mean_squared_error(y_valid_credits, y_pred_credits))
        return rmse
    
    def optimize(self, X_train, y_train, X_valid, y_valid, valid_credits_dangky, categorical_cols=None):
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER OPTIMIZATION FOR {self.model_type.upper()}")
        print(f"{'='*60}")
        print(f"Number of trials: {self.n_trials}")
        print(f"Timeout: {self.timeout} seconds")
        self.study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        self.study.optimize(
            lambda trial: self._objective(
                trial, X_train, y_train, X_valid, y_valid,
                valid_credits_dangky, categorical_cols
            ),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        self.best_params = self.study.best_params
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION COMPLETED!")
        print(f"{'='*60}")
        print(f"Best RMSE: {self.study.best_value:.4f}")
        print(f"Best parameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")
        return self.best_params
    
    def get_best_model(self):
        if self.best_params is None:
            raise ValueError("Optimization has not been run yet")
        full_params = self.best_params.copy()
        if self.model_type == 'xgboost':
            full_params.update({
                'n_estimators': 2000,
                'objective': 'reg:squarederror',
                'n_jobs': -1,
                'random_state': 42,
                'enable_categorical': True,
                'tree_method': 'hist',
            })
        
        return ModelTrainer(self.model_type, full_params)
    
    def plot_optimization_history(self, save_path=None):
        if self.study is None:
            raise ValueError("Optimization has not been run yet")
        import matplotlib.pyplot as plt
        fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Optimization history saved to: {save_path}")
        plt.tight_layout()
        plt.show()
    
    def plot_param_importances(self, save_path=None):
        """Plot parameter importances"""
        if self.study is None:
            raise ValueError("Optimization has not been run yet")
        import matplotlib.pyplot as plt
        fig = optuna.visualization.matplotlib.plot_param_importances(self.study)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter importances saved to: {save_path}")
        plt.tight_layout()
        plt.show()

def optimize_model(model_type, X_train, y_train, X_valid, y_valid,valid_credits_dangky, categorical_cols=None,n_trials=50, timeout=3600):
    optimizer = OptunaOptimizer(model_type, n_trials, timeout)
    best_params = optimizer.optimize(
        X_train, y_train, X_valid, y_valid,
        valid_credits_dangky, categorical_cols
    )
    best_model = optimizer.get_best_model()
    
    return best_params, best_model, optimizer