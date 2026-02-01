import optuna
from optuna.samplers import TPESampler
import numpy as np
from sklearn.model_selection import cross_val_score
from .models import ModelTrainer
from .config import RANDOM_STATE, OPTUNA_N_TRIALS, OPTUNA_TIMEOUT
from .evaluation import calculate_metrics


class OptunaOptimizer:    
    def __init__(self, model_type='xgboost', n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT):
        self.model_type = model_type
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params = None
        self.study = None
        
    def _objective(self, trial, X_train, y_train, X_valid, y_valid, categorical_cols):
        """Objective function for Optuna optimization"""
        
        if self.model_type == 'xgboost':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': RANDOM_STATE
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
                'random_state': RANDOM_STATE,
                'verbose': -1
            }
            
        elif self.model_type == 'catboost':
            params = {
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'iterations': trial.suggest_int('iterations', 100, 500),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_state': RANDOM_STATE,
                'verbose': False
            }
            
        elif self.model_type == 'random_forest':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'random_state': RANDOM_STATE
            }
        else:
            raise ValueError(f"Optimization not implemented for {self.model_type}")
        
        # Train model with these params
        trainer = ModelTrainer(self.model_type, params)
        trainer.train(X_train, y_train, X_valid, y_valid, categorical_cols)
        
        # Evaluate on validation set
        y_pred = trainer.predict(X_valid, categorical_cols)
        
        # Calculate RMSE (minimize this)
        metrics = calculate_metrics(y_valid, y_pred)
        rmse = metrics['RMSE']
        
        return rmse
    
    def optimize(self, X_train, y_train, X_valid, y_valid, categorical_cols=None):
        """Run hyperparameter optimization"""
        print(f"\n{'='*60}")
        print(f"Starting Hyperparameter Optimization for {self.model_type}")
        print(f"{'='*60}")
        print(f"Number of trials: {self.n_trials}")
        print(f"Timeout: {self.timeout} seconds")
        
        self.study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=RANDOM_STATE)
        )
        
        self.study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_valid, y_valid, categorical_cols),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        
        print(f"\n{'='*60}")
        print(f"Optimization Completed!")
        print(f"{'='*60}")
        print(f"Best RMSE: {self.study.best_value:.4f}")
        print(f"Best parameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")
        
        return self.best_params
    
    def get_best_model(self):
        """Get model with best parameters"""
        if self.best_params is None:
            raise ValueError("Optimization has not been run yet")
        
        return ModelTrainer(self.model_type, self.best_params)
    
    def plot_optimization_history(self, save_path=None):
        """Plot optimization history"""
        if self.study is None:
            raise ValueError("Optimization has not been run yet")
        
        import matplotlib.pyplot as plt
        
        fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Optimization history saved to: {save_path}")
        
        plt.close()
    
    def plot_param_importances(self, save_path=None):
        """Plot parameter importances"""
        if self.study is None:
            raise ValueError("Optimization has not been run yet")
        
        import matplotlib.pyplot as plt
        
        fig = optuna.visualization.matplotlib.plot_param_importances(self.study)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter importances saved to: {save_path}")
        
        plt.close()


def optimize_model(model_type, X_train, y_train, X_valid, y_valid, categorical_cols=None, 
                   n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT):
    """Convenience function to run optimization"""
    optimizer = OptunaOptimizer(model_type, n_trials, timeout)
    best_params = optimizer.optimize(X_train, y_train, X_valid, y_valid, categorical_cols)
    best_model = optimizer.get_best_model()
    
    return best_params, best_model, optimizer