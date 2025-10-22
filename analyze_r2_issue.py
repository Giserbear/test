import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['FontFamily'] = 'SimHei'

def analyze_data_quality(file_path):
    """分析数据质量，找出R²值偏小的原因"""
    try:
        df = pd.read_csv(file_path)
        print("=== 数据基本信息 ===")
        print(f"数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        
        # 检查各列的基本统计信息
        print("\n=== 各列统计信息 ===")
        for col in ['ALS_CHM', 'PRED_CHM', 'PRED_NEW_CHM']:
            if col in df.columns:
                print(f"\n{col}:")
                print(f"  非空值数量: {df[col].notna().sum()}")
                print(f"  缺失值数量: {df[col].isna().sum()}")
                print(f"  最小值: {df[col].min():.3f}")
                print(f"  最大值: {df[col].max():.3f}")
                print(f"  均值: {df[col].mean():.3f}")
                print(f"  标准差: {df[col].std():.3f}")
                print(f"  中位数: {df[col].median():.3f}")
        
        # 数据清洗
        true_data = df['ALS_CHM'].astype(float).values
        pred_1017 = df['PRED_CHM'].astype(float).values
        pred_1017_new = df['PRED_NEW_CHM'].astype(float).values
        
        # 分析有效数据
        print("\n=== 有效数据分析 ===")
        for name, pred_data in [("PRED_1017", pred_1017), ("PRED_NEW_CHM", pred_1017_new)]:
            mask = (~np.isnan(true_data)) & (~np.isnan(pred_data))
            x = true_data[mask]
            y = pred_data[mask]
            
            if len(x) > 0:
                print(f"\n{name} (有效样本数: {len(x)}):")
                
                # 计算相关系数
                corr = np.corrcoef(x, y)[0, 1]
                print(f"  皮尔逊相关系数: {corr:.4f}")
                
                # 计算R²
                r2 = r2_score(x, y)
                print(f"  R²: {r2:.4f}")
                
                # 计算RMSE
                rmse = np.sqrt(mean_squared_error(x, y))
                print(f"  RMSE: {rmse:.4f}")
                
                # 分析数据分布
                print(f"  真实值范围: [{x.min():.3f}, {x.max():.3f}]")
                print(f"  预测值范围: [{y.min():.3f}, {y.max():.3f}]")
                
                # 检查是否有异常值
                x_q1, x_q3 = np.percentile(x, [25, 75])
                y_q1, y_q3 = np.percentile(y, [25, 75])
                x_iqr = x_q3 - x_q1
                y_iqr = y_q3 - y_q1
                
                x_outliers = np.sum((x < x_q1 - 1.5*x_iqr) | (x > x_q3 + 1.5*x_iqr))
                y_outliers = np.sum((y < y_q1 - 1.5*y_iqr) | (y > y_q3 + 1.5*y_iqr))
                print(f"  真实值异常值数量: {x_outliers}")
                print(f"  预测值异常值数量: {y_outliers}")
                
                # 检查数据是否集中在某个范围
                print(f"  真实值在[0,5]范围内的比例: {np.mean((x >= 0) & (x <= 5)):.2%}")
                print(f"  真实值在[5,10]范围内的比例: {np.mean((x > 5) & (x <= 10)):.2%}")
                print(f"  真实值在[10,20]范围内的比例: {np.mean((x > 10) & (x <= 20)):.2%}")
                
                # 计算不同高度范围的R²
                ranges = [(0, 5), (5, 10), (10, 20), (20, 50)]
                for low, high in ranges:
                    range_mask = (x >= low) & (x < high)
                    if np.sum(range_mask) > 10:  # 至少10个样本
                        x_range = x[range_mask]
                        y_range = y[range_mask]
                        r2_range = r2_score(x_range, y_range)
                        print(f"  R²在[{low},{high})m范围: {r2_range:.4f} (样本数: {len(x_range)})")
        
        # 创建诊断图
        create_diagnostic_plots(true_data, pred_1017, pred_1017_new)
        
    except Exception as e:
        print(f"分析过程中出错: {e}")

def create_diagnostic_plots(true_data, pred_1017, pred_1017_new):
    """创建诊断图表"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 散点图对比
    for i, (pred_data, name) in enumerate([(pred_1017, 'PRED_1017'), (pred_1017_new, 'PRED_NEW_CHM')]):
        mask = (~np.isnan(true_data)) & (~np.isnan(pred_data))
        x = true_data[mask]
        y = pred_data[mask]
        
        if len(x) > 0:
            ax = axes[0, i]
            ax.scatter(x, y, alpha=0.6, s=20)
            
            # 添加拟合线
            slope, intercept = np.polyfit(x, y, 1)
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, 'r-', linewidth=2)
            
            # 1:1线
            diag_min = min(x.min(), y.min())
            diag_max = max(x.max(), y.max())
            ax.plot([diag_min, diag_max], [diag_min, diag_max], 'k--', alpha=0.7)
            
            r2 = r2_score(x, y)
            ax.set_title(f'{name}\nR² = {r2:.4f}')
            ax.set_xlabel('真实值')
            ax.set_ylabel('预测值')
            ax.grid(True, alpha=0.3)
    
    # 2. 残差图
    for i, (pred_data, name) in enumerate([(pred_1017, 'PRED_1017'), (pred_1017_new, 'PRED_NEW_CHM')]):
        mask = (~np.isnan(true_data)) & (~np.isnan(pred_data))
        x = true_data[mask]
        y = pred_data[mask]
        
        if len(x) > 0:
            ax = axes[1, i]
            residuals = y - x
            ax.scatter(x, residuals, alpha=0.6, s=20)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_title(f'{name} 残差图')
            ax.set_xlabel('真实值')
            ax.set_ylabel('残差 (预测值 - 真实值)')
            ax.grid(True, alpha=0.3)
    
    # 3. 数据分布直方图
    ax = axes[0, 2]
    mask = (~np.isnan(true_data)) & (~np.isnan(pred_1017))
    x = true_data[mask]
    y = pred_1017[mask]
    
    if len(x) > 0:
        ax.hist(x, bins=30, alpha=0.7, label='真实值', density=True)
        ax.hist(y, bins=30, alpha=0.7, label='预测值', density=True)
        ax.set_title('数据分布对比')
        ax.set_xlabel('高度值')
        ax.set_ylabel('密度')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. 残差分布
    ax = axes[1, 2]
    if len(x) > 0:
        residuals = y - x
        ax.hist(residuals, bins=30, alpha=0.7, density=True)
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_title('残差分布')
        ax.set_xlabel('残差值')
        ax.set_ylabel('密度')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = 'data/height_val/validate_chm_Max.csv'
    analyze_data_quality(file_path)