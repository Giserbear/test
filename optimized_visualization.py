import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set Chinese font for matplotlib
plt.rcParams['FontFamily'] = 'SimHei'
plt.rcParams['figure.dpi'] = 100


def load_and_prepare_data(file_path):
    """Load data and prepare arrays for analysis."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(df.head())
        
        # Prepare data arrays
        true_data = df['ALS_CHM'].astype(float).values
        pred_1017 = df['PRED_CHM'].astype(float).values
        pred_1017_new = df['PRED_NEW_CHM'].astype(float).values
        
        return true_data, pred_1017, pred_1017_new
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None


def calculate_metrics(true_values, pred_values):
    """Calculate comprehensive evaluation metrics."""
    # Remove NaN values
    mask = (~np.isnan(true_values)) & (~np.isnan(pred_values))
    true_clean = true_values[mask]
    pred_clean = pred_values[mask]
    
    if len(true_clean) == 0:
        return None
    
    # Calculate metrics
    r2 = r2_score(true_clean, pred_clean)
    rmse = np.sqrt(mean_squared_error(true_clean, pred_clean))
    
    # MAPE calculation with zero handling
    nonzero_mask = true_clean != 0
    if nonzero_mask.any():
        mape = np.mean(np.abs((true_clean[nonzero_mask] - pred_clean[nonzero_mask]) / true_clean[nonzero_mask])) * 100
    else:
        mape = np.nan
    
    bias = np.mean(pred_clean - true_clean)
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mape': mape,
        'bias': bias,
        'n_samples': len(true_clean)
    }


def plot_comparison(true_data, pred_data_list, labels, figsize=(12, 5)):
    """Create comparison plots for multiple prediction methods."""
    n_plots = len(pred_data_list)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    # Handle single plot case
    if n_plots == 1:
        axes = [axes]
    
    for ax, pred_data, label in zip(axes, pred_data_list, labels):
        # Clean data
        mask = (~np.isnan(true_data)) & (~np.isnan(pred_data))
        x = true_data[mask]
        y = pred_data[mask]
        
        if len(x) == 0:
            ax.text(0.5, 0.5, '无有效样本', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{label} - 无数据')
            continue
        
        # Calculate metrics
        metrics = calculate_metrics(x, y)
        if metrics is None:
            continue
        
        # Scatter plot
        ax.scatter(x, y, alpha=0.6, s=30, color='tab:blue')
        
        # Linear regression line
        slope, intercept = np.polyfit(x, y, 1)
        x_sorted = np.sort(x)
        y_fit = slope * x_sorted + intercept
        ax.plot(x_sorted, y_fit, color='red', linewidth=2,
                label=f'拟合: y={slope:.3f}x+{intercept:.3f}')
        
        # 1:1 reference line
        diag_min = min(x.min(), y.min())
        diag_max = max(x.max(), y.max())
        ax.plot([diag_min, diag_max], [diag_min, diag_max], 'k--', linewidth=1, label='1:1')
        
        # Labels and title
        ax.set_xlabel('真实 ALS_CHM (m)', fontsize=12)
        ax.set_ylabel(f'预测 {label} (m)', fontsize=12)
        ax.set_title(f'{label}\nR²={metrics["r2"]:.3f}, RMSE={metrics["rmse"]:.3f}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        # Set equal aspect ratio for better comparison
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig


def plot_detailed_analysis(true_data, pred_data, title_suffix="", figsize=(6, 6)):
    """Create detailed analysis plot with comprehensive metrics."""
    # Clean data
    mask = (~np.isnan(true_data)) & (~np.isnan(pred_data))
    x = true_data[mask]
    y = pred_data[mask]
    
    if len(x) == 0:
        print("No valid data for detailed analysis")
        return None
    
    # Calculate metrics
    metrics = calculate_metrics(x, y)
    if metrics is None:
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(x, y, alpha=0.6, s=30, color='tab:blue', label='样本点')
    
    # 1:1 reference line
    diag_min = 0
    diag_max = 20
    ax.plot([diag_min, diag_max], [diag_min, diag_max], 'k--', linewidth=1.5, label='1:1')
    
    # Linear regression line
    slope, intercept = np.polyfit(x, y, 1)
    x_range = np.linspace(0, 20, 100)
    y_fit = slope * x_range + intercept
    ax.plot(x_range, y_fit, color='red', linewidth=2, alpha=0.8, label=f'拟合线')
    
    # Labels and formatting
    ax.set_xlabel('真实高度值(m)', fontsize=14)
    ax.set_ylabel('预测高度值(m)', fontsize=14)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_xticks(np.arange(0, 21, 5))
    ax.set_yticks(np.arange(0, 21, 5))
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(alpha=0.3)
    
    # Metrics text box
    textstr = (f'R² = {metrics["r2"]:.3f}\n'
               f'RMSE = {metrics["rmse"]:.3f}\n'
               f'MAPE = {metrics["mape"]:.2f}%\n'
               f'Bias = {metrics["bias"]:.3f}\n'
               f'样本数 = {metrics["n_samples"]}')
    
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    
    return fig


def main():
    """Main execution function."""
    # File path
    file_path = 'data/height_val/validate_chm_Max.csv'
    
    # Load data
    true_data, pred_1017, pred_1017_new = load_and_prepare_data(file_path)
    
    if true_data is None:
        print("Failed to load data. Exiting.")
        return
    
    # Create comparison plots
    print("\n创建对比图...")
    fig1 = plot_comparison(true_data, [pred_1017, pred_1017_new], 
                          ['Pred_1017', 'Pred_1017_new'])
    plt.show()
    
    # Create detailed analysis for first prediction
    print("\n创建详细分析图...")
    fig2 = plot_detailed_analysis(true_data, pred_1017, "Pred_1017")
    if fig2 is not None:
        plt.show()
    
    # Print summary statistics
    print("\n=== 评估指标总结 ===")
    for name, pred_data in [("Pred_1017", pred_1017), ("Pred_1017_new", pred_1017_new)]:
        metrics = calculate_metrics(true_data, pred_data)
        if metrics:
            print(f"\n{name}:")
            print(f"  R² = {metrics['r2']:.4f}")
            print(f"  RMSE = {metrics['rmse']:.4f}")
            print(f"  MAPE = {metrics['mape']:.2f}%")
            print(f"  Bias = {metrics['bias']:.4f}")
            print(f"  有效样本数 = {metrics['n_samples']}")
    
    print("\n分析完成!")


if __name__ == "__main__":
    main()