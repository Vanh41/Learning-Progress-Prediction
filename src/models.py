import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import catboost as cb


class ModelTrainer:    
    def __init__(self, model_type='xgboost', params=None):
        self.model_type = model_type
        self.params = params if params else self._get_default_params()
        self.model = None
        self.label_encoders = {}
        
    def _get_default_params(self):
        RANDOM_STATE = 42
        
        defaults = {
            'xgboost': {
                'n_estimators': 2000,
                'learning_rate': 0.02,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'n_jobs': -1,
                'random_state': RANDOM_STATE,
                'enable_categorical': True,  
                'tree_method': 'hist',      
            },
            'lightgbm': {
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.5,
                'reg_lambda': 1.0,
                'random_state': RANDOM_STATE,
                'n_jobs': -1,
                'verbose': -1
            },
            'catboost': {
                'depth': 6,
                'learning_rate': 0.05,
                'iterations': 200,
                'l2_leaf_reg': 3.0,
                'random_state': RANDOM_STATE,
                'verbose': False,
                'thread_count': -1
            }
        }
        
        return defaults.get(self.model_type, {})
    
    def _encode_categorical_features(self, X, categorical_cols=None, is_training=True):
        X = X.copy()
        
        # Get columns to encode
        object_cols = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            cols_to_encode = list(set(object_cols) | set(categorical_cols))
        else:
            cols_to_encode = object_cols
        
        for col in cols_to_encode:
            if col not in X.columns:
                continue
                
            X[col] = X[col].astype(str).fillna('UNKNOWN')
            
            if is_training:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unknown categories
                    X[col] = X[col].map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    X[col] = -1
        
        return X
    
    def train(self, X_train, y_train, X_valid=None, y_valid=None, categorical_cols=None):
        if categorical_cols is None:
            categorical_cols = []
        
        print(f"\n[Training {self.model_type}]")
        print(f"  Training samples: {len(X_train)}")
        if categorical_cols:
            print(f"  Categorical features: {categorical_cols}")
        if self.model_type == 'xgboost':
            X_train_enc = self._encode_categorical_features(X_train, categorical_cols, is_training=True)

            X_valid_enc = None
            if X_valid is not None:
                X_valid_enc = self._encode_categorical_features(
                    X_valid, categorical_cols, is_training=False
                )

            self.model = xgb.XGBRegressor(**self.params)

            if X_valid is not None and y_valid is not None:
                self.model.fit(
                    X_train_enc, y_train,
                    eval_set=[(X_valid_enc, y_valid)],
                    verbose=200
                )
            else:
                self.model.fit(X_train_enc, y_train)
        
        # LightGBM - Tự động nhận diện category dtype
        elif self.model_type == 'lightgbm':
            X_train_encoded = self._encode_categorical_features(
                X_train, categorical_cols, is_training=True
            )
            X_valid_encoded = None
            if X_valid is not None:
                X_valid_encoded = self._encode_categorical_features(
                    X_valid, categorical_cols, is_training=False
                )
            
            self.model = lgb.LGBMRegressor(**self.params)
            
            if X_valid is not None and y_valid is not None:
                self.model.fit(
                    X_train_encoded, y_train,
                    eval_set=[(X_train_encoded, y_train), (X_valid_encoded, y_valid)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
                )
            else:
                self.model.fit(X_train_encoded, y_train)
        
        # CatBoost - Hỗ trợ categorical natively
        elif self.model_type == 'catboost':
            cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            cat_features = [i for i, col in enumerate(X_train.columns) if col in cat_cols]
            
            X_train_cb = X_train.copy()
            for col in cat_cols:
                X_train_cb[col] = X_train_cb[col].astype(str).fillna('UNKNOWN')
            
            X_valid_cb = None
            if X_valid is not None:
                X_valid_cb = X_valid.copy()
                for col in cat_cols:
                    X_valid_cb[col] = X_valid_cb[col].astype(str).fillna('UNKNOWN')
            
            self.model = cb.CatBoostRegressor(**self.params)
            
            if X_valid is not None and y_valid is not None:
                self.model.fit(
                    X_train_cb, y_train,
                    cat_features=cat_features,
                    eval_set=(X_valid_cb, y_valid),
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                self.model.fit(
                    X_train_cb, y_train,
                    cat_features=cat_features,
                    verbose=False
                )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"  {self.model_type} model trained successfully")
        return self.model
    
    def predict(self, X, categorical_cols=None):
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if categorical_cols is None:
            categorical_cols = []
        
        # XGBoost
        if self.model_type == 'xgboost':
            X_pred = self._encode_categorical_features(X, categorical_cols, is_training=False)
            predictions = self.model.predict(X_pred)
        
        # CatBoost
        elif self.model_type == 'catboost':
            X_pred = X.copy()
            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in cat_cols:
                X_pred[col] = X_pred[col].astype(str).fillna('UNKNOWN')
            predictions = self.model.predict(X_pred)
        
        # Other models (need encoding)
        else:
            X_encoded = self._encode_categorical_features(
                X, categorical_cols, is_training=False
            )
            predictions = self.model.predict(X_encoded)
        
        # Post-processing: Clip predictions
        predictions = np.maximum(predictions, 0)
        
        # If TC_DANGKY is in X, clip to max capacity
        if 'TC_DANGKY' in X.columns:
            predictions = np.minimum(predictions, X['TC_DANGKY'].values)
        
        return predictions
    
    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_)
        else:
            return None


class EnsembleModel:    
    def __init__(self, models_config=None):
        self.models = []
        self.weights = []
        
        if models_config is None:
            # Default ensemble config
            models_config = [
                ('xgboost', None, 0.4),
                ('lightgbm', None, 0.3),
                ('catboost', None, 0.3),
            ]
        
        for model_type, params, weight in models_config:
            self.models.append(ModelTrainer(model_type, params))
            self.weights.append(weight)
        
        self.weights = np.array(self.weights) / sum(self.weights)
        
    def train(self, X_train, y_train, X_valid=None, y_valid=None, categorical_cols=None):
        """Train all models in ensemble"""
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{len(self.models)}: {model.model_type}")
            model.train(X_train, y_train, X_valid, y_valid, categorical_cols)
        print("\n All models trained successfully")
        
    def predict(self, X, categorical_cols=None):
        """Make weighted ensemble predictions"""
        predictions = np.zeros(len(X))
        
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X, categorical_cols)
            predictions += pred * weight
        
        # Post-processing
        predictions = np.maximum(predictions, 0)
        if 'TC_DANGKY' in X.columns:
            predictions = np.minimum(predictions, X['TC_DANGKY'].values)
            
        return predictions

    def get_individual_predictions(self, X, categorical_cols=None):
        """Get predictions from each model separately"""
        all_preds = {}
        for model in self.models:
            all_preds[model.model_type] = model.predict(X, categorical_cols)
        return all_preds


def create_model(model_type='xgboost', params=None):
    """Factory function to create a model trainer"""
    return ModelTrainer(model_type, params)


def create_ensemble(models_config=None):
    """Factory function to create an ensemble model"""
    return EnsembleModel(models_config)