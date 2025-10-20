import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['FontFamily'] = 'SimHei'

def comprehensive_r2_analysis(true_data, pred_data, data_name="数据"):
    """
    全面分析R²值偏小的原因
    """
    print(f"\n=== {data_name} R² 诊断分析 ===")
    
    # 数据清洗
    mask = (~np.isnan(true_data)) & (~np.isnan(pred_data))
    x = true_data[mask]
    y = pred_data[mask]
    
    if len(x) == 0:
        print("无有效数据")
        return
    
    print(f"有效样本数: {len(x)}")
    
    # 1. 基本统计信息
    print(f"\n1. 基本统计信息:")
    print(f"   真实值: 均值={x.mean():.3f}, 标准差={x.std():.3f}, 范围=[{x.min():.3f}, {x.max():.3f}]")
    print(f"   预测值: 均值={y.mean():.3f}, 标准差={y.std():.3f}, 范围=[{y.min():.3f}, {y.max():.3f}]")
    
    # 2. 相关性分析
    corr_pearson = np.corrcoef(x, y)[0, 1]
    corr_spearman = stats.spearmanr(x, y)[0]
    print(f"\n2. 相关性分析:")
    print(f"   皮尔逊相关系数: {corr_pearson:.4f}")
    print(f"   斯皮尔曼相关系数: {corr_spearman:.4f}")
    
    # 3. R² 计算
    r2 = r2_score(x, y)
    print(f"\n3. R² 分析:")
    print(f"   R² = {r2:.4f}")
    
    # 4. 方差分析
    ss_tot = np.sum((x - x.mean()) ** 2)  # 总平方和
    ss_res = np.sum((x - y) ** 2)         # 残差平方和
    ss_reg = np.sum((y - x.mean()) ** 2)  # 回归平方和
    
    print(f"\n4. 方差分解:")
    print(f"   总平方和 (SST): {ss_tot:.2f}")
    print(f"   残差平方和 (SSE): {ss_res:.2f}")
    print(f"   回归平方和 (SSR): {ss_reg:.2f}")
    print(f"   R² = 1 - SSE/SST = 1 - {ss_res:.2f}/{ss_tot:.2f} = {1 - ss_res/ss_tot:.4f}")
    
    # 5. 残差分析
    residuals = y - x
    print(f"\n5. 残差分析:")
    print(f"   残差均值: {residuals.mean():.4f}")
    print(f"   残差标准差: {residuals.std():.4f}")
    print(f"   残差范围: [{residuals.min():.4f}, {residuals.max():.4f}]")
    
    # 6. 不同高度范围的R²
    print(f"\n6. 分段R²分析:")
    ranges = [(0, 2), (2, 5), (5, 10), (10, 15), (15, 25)]
    for low, high in ranges:
        range_mask = (x >= low) & (x < high)
        if np.sum(range_mask) > 5:  # 至少5个样本
            x_range = x[range_mask]
            y_range = y[range_mask]
            r2_range = r2_score(x_range, y_range)
            corr_range = np.corrcoef(x_range, y_range)[0, 1]
            print(f"   [{low:2d}, {high:2d})m: R²={r2_range:.4f}, 相关系数={corr_range:.4f}, 样本数={len(x_range)}")
    
    # 7. 异常值检测
    print(f"\n7. 异常值检测:")
    
    # 基于残差的异常值
    residual_std = residuals.std()
    outlier_mask = np.abs(residuals) > 3 * residual_std
    n_outliers = np.sum(outlier_mask)
    print(f"   基于残差的异常值 (>3σ): {n_outliers} 个 ({n_outliers/len(x)*100:.1f}%)")
    
    if n_outliers > 0:
        # 移除异常值后重新计算R²
        x_clean = x[~outlier_mask]
        y_clean = y[~outlier_mask]
        if len(x_clean) > 5:
            r2_clean = r2_score(x_clean, y_clean)
            print(f"   移除异常值后R²: {r2_clean:.4f}")
    
    # 8. 数据分布分析
    print(f"\n8. 数据分布分析:")
    print(f"   真实值变异系数 (CV): {x.std()/x.mean()*100:.2f}%")
    print(f"   预测值变异系数 (CV): {y.std()/y.mean()*100:.2f}%")
    
    # 9. 线性回归分析
    print(f"\n9. 线性回归分析:")
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(f"   回归方程: y = {slope:.4f}x + {intercept:.4f}")
    print(f"   回归R²: {r_value**2:.4f}")
    print(f"   回归p值: {p_value:.2e}")
    
    # 10. 建议
    print(f"\n10. 改进建议:")
    if r2 < 0.3:
        print("   - R² < 0.3: 模型预测能力较差，建议检查模型算法")
    elif r2 < 0.6:
        print("   - R² < 0.6: 模型预测能力中等，建议优化特征工程")
    else:
        print("   - R² >= 0.6: 模型预测能力较好")
    
    if corr_pearson < 0.5:
        print("   - 相关系数较低，可能存在非线性关系")
    
    if n_outliers > len(x) * 0.1:
        print("   - 异常值较多，建议进行数据清洗")
    
    if abs(residuals.mean()) > 0.1:
        print("   - 存在系统性偏差，建议调整模型")
    
    return {
        'r2': r2,
        'correlation': corr_pearson,
        'n_samples': len(x),
        'residuals_mean': residuals.mean(),
        'residuals_std': residuals.std(),
        'n_outliers': n_outliers
    }

def create_diagnostic_plots(true_data, pred_data, data_name="数据"):
    """创建诊断图表"""
    mask = (~np.isnan(true_data)) & (~np.isnan(pred_data))
    x = true_data[mask]
    y = pred_data[mask]
    
    if len(x) == 0:
        print("无有效数据用于绘图")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 散点图 + 拟合线
    ax = axes[0, 0]
    ax.scatter(x, y, alpha=0.6, s=20)
    
    # 线性拟合
    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'拟合线: y={slope:.3f}x+{intercept:.3f}')
    
    # 1:1线
    diag_min = min(x.min(), y.min())
    diag_max = max(x.max(), y.max())
    ax.plot([diag_min, diag_max], [diag_min, diag_max], 'k--', alpha=0.7, label='1:1线')
    
    r2 = r2_score(x, y)
    ax.set_title(f'{data_name} - R² = {r2:.4f}')
    ax.set_xlabel('真实值')
    ax.set_ylabel('预测值')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 残差图
    ax = axes[0, 1]
    residuals = y - x
    ax.scatter(x, residuals, alpha=0.6, s=20)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_title(f'{data_name} - 残差图')
    ax.set_xlabel('真实值')
    ax.set_ylabel('残差 (预测值 - 真实值)')
    ax.grid(True, alpha=0.3)
    
    # 3. 残差分布
    ax = axes[1, 0]
    ax.hist(residuals, bins=30, alpha=0.7, density=True)
    ax.axvline(x=0, color='r', linestyle='--')
    ax.set_title(f'{data_name} - 残差分布')
    ax.set_xlabel('残差值')
    ax.set_ylabel('密度')
    ax.grid(True, alpha=0.3)
    
    # 4. Q-Q图 (正态性检验)
    ax = axes[1, 1]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(f'{data_name} - 残差Q-Q图')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 使用示例
def analyze_your_data():
    """
    分析您的数据 - 请替换为实际的数据加载代码
    """
    # 这里需要您提供实际的数据文件路径
    # df = pd.read_csv('data/height_val/validate_chm_Max.csv')
    # true_data = df['ALS_CHM'].astype(float).values
    # pred_1017 = df['PRED_CHM'].astype(float).values
    # pred_1017_new = df['PRED_NEW_CHM'].astype(float).values
    
    print("请将您的数据文件路径替换到 analyze_your_data() 函数中")
    print("然后运行以下代码进行分析：")
    print("""
    # 加载数据
    df = pd.read_csv('data/height_val/validate_chm_Max.csv')
    true_data = df['ALS_CHM'].astype(float).values
    pred_1017 = df['PRED_CHM'].astype(float).values
    pred_1017_new = df['PRED_NEW_CHM'].astype(float).values
    
    # 分析R²问题
    comprehensive_r2_analysis(true_data, pred_1017, "PRED_1017")
    comprehensive_r2_analysis(true_data, pred_1017_new, "PRED_1017_NEW")
    
    # 创建诊断图
    create_diagnostic_plots(true_data, pred_1017, "PRED_1017")
    create_diagnostic_plots(true_data, pred_1017_new, "PRED_1017_NEW")
    """)

if __name__ == "__main__":
    analyze_your_data()