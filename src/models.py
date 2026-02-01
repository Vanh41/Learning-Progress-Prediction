import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from .config import RANDOM_STATE, DEFAULT_PARAMS, EARLY_STOPPING_ROUNDS
from .utils import set_seed


class ModelTrainer:    
    def __init__(self, model_type='xgboost', params=None):
        self.model_type = model_type
        self.params = params if params else DEFAULT_PARAMS.get(model_type, {})
        self.model = None
        self.label_encoders = {}
        
        set_seed(RANDOM_STATE)
        
    def _encode_categorical_features(self, X, categorical_cols, is_training=True):
        X = X.copy()
        
        for col in categorical_cols:
            if col in X.columns:
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    if col in self.label_encoders:
                        known_categories = set(self.label_encoders[col].classes_)
                        X[col] = X[col].astype(str).apply(
                            lambda x: x if x in known_categories else 'Unknown'
                        )
                        if 'Unknown' not in known_categories:
                            self.label_encoders[col].classes_ = np.append(
                                self.label_encoders[col].classes_, 'Unknown'
                            )
                        X[col] = self.label_encoders[col].transform(X[col])
                    else:
                        X[col] = 0
        
        return X
    
    def train(self, X_train, y_train, X_valid=None, y_valid=None, categorical_cols=None):
        if categorical_cols is None:
            categorical_cols = []
        
        print(f"\n[Training {self.model_type}]")
        print(f"  Training samples: {len(X_train)}")
        
        # Encode categorical features (except for CatBoost)
        if self.model_type != 'catboost':
            X_train_encoded = self._encode_categorical_features(X_train, categorical_cols, is_training=True)
            if X_valid is not None:
                X_valid_encoded = self._encode_categorical_features(X_valid, categorical_cols, is_training=False)
        else:
            X_train_encoded = X_train.copy()
            X_valid_encoded = X_valid.copy() if X_valid is not None else None
        
        # Train model logic
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(**self.params)
            if X_valid is not None:
                self.model.fit(X_train_encoded, y_train, eval_set=[(X_valid_encoded, y_valid)], verbose=False)
            else:
                self.model.fit(X_train_encoded, y_train)
                
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(**self.params)
            if X_valid is not None:
                self.model.fit(
                    X_train_encoded, y_train,
                    eval_set=[(X_train_encoded, y_train), (X_valid_encoded, y_valid)],
                    callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS), lgb.log_evaluation(0)]
                )
            else:
                self.model.fit(X_train_encoded, y_train)
                
        elif self.model_type == 'catboost':
            cat_features = [i for i, col in enumerate(X_train.columns) if col in categorical_cols]
            self.model = cb.CatBoostRegressor(**self.params)
            if X_valid is not None:
                self.model.fit(X_train_encoded, y_train, cat_features=cat_features, 
                               eval_set=(X_valid_encoded, y_valid), 
                               early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
            else:
                self.model.fit(X_train_encoded, y_train, cat_features=cat_features, verbose=False)
                
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(**self.params)
            self.model.fit(X_train_encoded, y_train)
            
        elif self.model_type == 'linear':
            self.model = LinearRegression()
            self.model.fit(X_train_encoded, y_train)
            
        elif self.model_type == 'ridge':
            self.model = Ridge(**self.params)
            self.model.fit(X_train_encoded, y_train)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"  {self.model_type} model trained successfully")
        return self.model
    
    def predict(self, X, categorical_cols=None):
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if categorical_cols is None:
            categorical_cols = []
        
        if self.model_type == 'catboost':
            X_encoded = X.copy()
        else:
            X_encoded = self._encode_categorical_features(X, categorical_cols, is_training=False)
        predictions = self.model.predict(X_encoded)
        predictions = np.maximum(predictions, 0)
        if 'TC_DANGKY' in X.columns:
            predictions = np.minimum(predictions, X['TC_DANGKY'].values)
            
        return predictions
    
    def get_feature_importance(self):
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
            models_config = [
                ('xgboost', DEFAULT_PARAMS['xgboost'], 0.4),
                ('lightgbm', DEFAULT_PARAMS['lightgbm'], 0.3),
                ('catboost', DEFAULT_PARAMS['catboost'], 0.3),
            ]
        
        for model_type, params, weight in models_config:
            self.models.append(ModelTrainer(model_type, params))
            self.weights.append(weight)
        
        self.weights = np.array(self.weights) / sum(self.weights)
        
    def train(self, X_train, y_train, X_valid=None, y_valid=None, categorical_cols=None):
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{len(self.models)}: {model.model_type}")
            model.train(X_train, y_train, X_valid, y_valid, categorical_cols)
        print("\nAll models trained successfully")
        
    def predict(self, X, categorical_cols=None):
        predictions = np.zeros(len(X))
        
        for model, weight in zip(self.models, self.weights):
            # ModelTrainer.predict đã có chặn trên/dưới nhưng ta vẫn nên đảm bảo sau khi cộng trọng số
            pred = model.predict(X, categorical_cols)
            predictions += pred * weight
        
        # Đảm bảo kết quả ensemble cuối cùng vẫn tuân thủ logic
        predictions = np.maximum(predictions, 0)
        if 'TC_DANGKY' in X.columns:
            predictions = np.minimum(predictions, X['TC_DANGKY'].values)
            
        return predictions # Trả về số thực

    def get_individual_predictions(self, X, categorical_cols=None):
        all_preds = {}
        for i, model in enumerate(self.models):
            all_preds[model.model_type] = model.predict(X, categorical_cols)
        return all_preds

def create_model(model_type='xgboost', params=None):
    return ModelTrainer(model_type, params)

def create_ensemble(models_config=None):
    return EnsembleModel(models_config)