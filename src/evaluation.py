import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path


def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError(
            f"Empty input: y_true size={y_true.size}, y_pred size={y_pred.size}"
        )
    
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0
    
    metrics = {
        'R2': r2,
        'RMSE': rmse,
        'MSE': mse,
        'MAE': mae,
        'MAPE': mape
    }
    
    return metrics


def print_metrics(metrics, title="Model Performance"):
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    for metric, value in metrics.items():
        print(f"{metric:10s}: {value:.4f}")
    print(f"{'='*50}\n")


def plot_predictions(y_true, y_pred, title="Predictions vs Actual", save_path=None):
    plt.figure(figsize=(10, 6))
    
    plt.scatter(y_true, y_pred, alpha=0.5, s=30)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual TC_HOANTHANH', fontsize=12)
    plt.ylabel('Predicted TC_HOANTHANH', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred, title="Residual Plot", save_path=None):
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=30)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title('Residuals vs Predicted', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Residuals', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20, title="Feature Importance", save_path=None):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        print("Model does not have feature importance attribute")
        return
    
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_imp_df.head(top_n), y='feature', x='importance', palette='viridis')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3, axis='x')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    return feature_imp_df


def plot_error_distribution_by_groups(y_true, y_pred, group_labels, title="Error by Groups", save_path=None):
    errors = y_true - y_pred
    
    df = pd.DataFrame({
        'error': errors,
        'group': group_labels
    })
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='group', y='error', palette='Set2')
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Group', fontsize=12)
    plt.ylabel('Prediction Error', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()


def create_evaluation_report(
    y_true,
    y_pred,
    feature_names=None,
    model=None,
    save_dir=None
):
    from .config import OUTPUT_DIR
    import numpy as np

    if save_dir is None:
        save_dir = OUTPUT_DIR / 'evaluation'
        save_dir.mkdir(exist_ok=True, parents=True)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.size == 0 or y_pred.size == 0:
        print("[Evaluation] Empty y_true or y_pred. Skipping evaluation report.")
        print(f"    y_true size: {y_true.size}, y_pred size: {y_pred.size}")
        return None

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )

    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics, title="Evaluation Metrics")

    plot_predictions(
        y_true,
        y_pred,
        title="Predictions vs Actual Values",
        save_path=save_dir / 'predictions_vs_actual.png'
    )

    plot_residuals(
        y_true,
        y_pred,
        title="Residual Analysis",
        save_path=save_dir / 'residuals.png'
    )

    if model is not None and feature_names is not None:
        if len(feature_names) == 0:
            print("Feature names empty. Skipping feature importance.")
        else:
            feature_imp_df = plot_feature_importance(
                model,
                feature_names,
                title="Top 20 Important Features",
                save_path=save_dir / 'feature_importance.png'
            )
            feature_imp_df.to_csv(
                save_dir / 'feature_importance.csv',
                index=False
            )

    return metrics



def analyze_predictions_by_segments(y_true, y_pred, X, segment_col, save_dir=None):
    from .config import OUTPUT_DIR
    
    if save_dir is None:
        save_dir = OUTPUT_DIR / 'evaluation'
        save_dir.mkdir(exist_ok=True)
    
    results = []
    
    for segment in X[segment_col].unique():
        mask = X[segment_col] == segment
        
        if mask.sum() > 0:
            segment_metrics = calculate_metrics(y_true[mask], y_pred[mask])
            segment_metrics['segment'] = segment
            segment_metrics['count'] = mask.sum()
            results.append(segment_metrics)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('RMSE')
    
    print("\nMetrics by Segment:")
    print(results_df.to_string(index=False))
    
    results_df.to_csv(save_dir / f'metrics_by_{segment_col}.csv', index=False)
    
    return results_df