import re
import matplotlib.pyplot as plt
import numpy as np

# ================= 配置区域 =================
# 6大模型对应的日志文件路径
# 请确保这些文件都在当前目录下，或者修改为绝对路径
LOG_FILES = {
    'SF': 'log_single_frame.txt',
    'TG-Scratch': 'log_overfit.txt',
    'TG-Nofrozen': 'loso_overfit_nofrozen.txt',
    # '1D-CNN': 'log_cnn.txt',
    # 'LSTM': 'log_lstm.txt',
    'TG': 'log_gru.txt'

}

SAVE_NAME = 'final_4_temporal_models_comparison.png'

# 绘图样式配置
plt.rcParams['font.family'] = 'serif'  # 论文标准衬线字体
# 如果系统有 Times New Roman，可以取消下面这行的注释
# plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 24  # 增大一倍 (原12)
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.weight'] = 'bold'  # 全局加粗


# ================= 工具函数 =================

def parse_metrics(file_path):
    """
    从日志文件的 'LOSO-CV FINAL RESULT' 部分解析 Mean 和 Std
    """
    metrics = {}
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # 寻找最终结果段落
            if "LOSO-CV FINAL RESULT" in content:
                # 解析 accuracy 行: accuracy       98.7408 0.5607
                acc_match = re.search(r'accuracy\s+(\d+\.\d+)\s+(\d+\.\d+)', content)
                if acc_match:
                    metrics['acc_mean'] = float(acc_match.group(1))
                    metrics['acc_std'] = float(acc_match.group(2))

                # 解析 fall_recall 行: fall_recall    93.0012 4.3201
                rec_match = re.search(r'fall_recall\s+(\d+\.\d+)\s+(\d+\.\d+)', content)
                if rec_match:
                    metrics['rec_mean'] = float(rec_match.group(1))
                    metrics['rec_std'] = float(rec_match.group(2))
    except Exception as e:
        print(f"Warning: 读取文件 {file_path} 失败: {e}")
    return metrics


def get_capped_errors(means, stds):
    """
    计算非对称误差棒：
    - 下限：Mean - Std (保持原样)
    - 上限：Mean + Std (但在 100% 处截断，不画出界)
    返回格式: [lower_errors, upper_errors]
    """
    lower_errors = stds
    # 计算上限：取 Std 和 (100 - Mean) 中的较小值
    # 这样 Mean + upper_error 永远不会超过 100
    upper_errors = np.minimum(stds, 100 - means)
    return np.array([lower_errors, upper_errors])


# ================= 主逻辑 =================

def plot_final_comparison():
    # 1. 读取数据
    print("正在解析日志文件...")
    data = {}
    for model_name, path in LOG_FILES.items():
        res = parse_metrics(path)
        if not res:
            print(f"  [Error] 无法从 {path} 解析数据，将使用 0 填充。")
        data[model_name] = res

    # 2. 准备绘图列表
    models = list(LOG_FILES.keys())
    # 使用 NumPy 数组以便进行向量化计算
    acc_means = np.array([data[m].get('acc_mean', 0) for m in models])
    acc_stds = np.array([data[m].get('acc_std', 0) for m in models])

    rec_means = np.array([data[m].get('rec_mean', 0) for m in models])
    rec_stds = np.array([data[m].get('rec_std', 0) for m in models])

    # 3. 计算截断后的误差棒
    acc_yerr = get_capped_errors(acc_means, acc_stds)
    rec_yerr = get_capped_errors(rec_means, rec_stds)

    # 4. 开始绘图
    # 保持图表尺寸不变 (14, 7)
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(models))
    width = 0.35  # 保持柱体宽度不变

    # 绘制 Accuracy 柱子 (蓝色)
    rects1 = ax.bar(x - width / 2, acc_means, width, yerr=acc_yerr,
                    label='Overall Accuracy', color='#4F81BD',
                    capsize=5, edgecolor='black', linewidth=1.2, hatch='//',
                    error_kw={'elinewidth': 1.5, 'ecolor': 'black'})

    # 绘制 Fall Recall 柱子 (红色)
    rects2 = ax.bar(x + width / 2, rec_means, width, yerr=rec_yerr,
                    label='Fall Recall (Sensitivity)', color='#C0504D',
                    capsize=5, edgecolor='black', linewidth=1.2, hatch='..',
                    error_kw={'elinewidth': 1.5, 'ecolor': 'black'})

    # 5. 标注数值
    # fontsize 10 -> 20, fontweight 保持 bold
    ax.bar_label(rects1, padding=3, fmt='%.1f', fontsize=20, fontweight='bold')
    ax.bar_label(rects2, padding=3, fmt='%.1f', fontsize=20, fontweight='bold')

    # 6. 美化图表
    # ylabel fontsize 14 -> 28
    ax.set_ylabel('Performance (%)', fontsize=28, fontweight='bold')

    # title fontsize 16 -> 32
    # ax.set_title('Performance Comparison of Temporal Modeling Architectures', fontsize=32, pad=20, fontweight='bold')

    ax.set_xticks(x)
    # xticklabels fontsize 12 -> 24
    ax.set_xticklabels(models, fontsize=24, fontweight='bold')

    # 图例放底部
    # legend fontsize 12 -> 24, prop={'weight':'bold'}
    ax.legend(loc='lower center', fontsize=24, frameon=True, edgecolor='black', ncol=2,
              prop={'weight': 'bold', 'size': 24})

    # 动态调整 Y 轴下限，上限固定留出空间给标签
    # 自动寻找数据的最低点（含误差棒下沿）
    min_point = min(
        (acc_means - acc_stds).min(),
        (rec_means - rec_stds).min()
    )
    # 下限设为最低点再减 2%，上限设为 103% (为了放数字)
    ax.set_ylim(bottom=max(0, min_point - 2), top=103)

    # 去除上方和右侧边框 (学术风格)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 添加网格线
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

    # 因为字体变大了，layout 可能会比较挤，调整 rect 来留出更多空间
    # 或者直接 tight_layout, 但 title 可能会被切掉如果太大
    plt.tight_layout()
    plt.savefig(SAVE_NAME, dpi=300)
    print(f"✅ 绘图完成！图片已保存至: {SAVE_NAME}")

    # 打印最终数据表供核对
    print("\n=== Final Data Table (Mean ± Std) ===")
    print(f"{'Model':<20} | {'Accuracy':<15} | {'Fall Recall':<15}")
    print("-" * 56)
    for i, m in enumerate(models):
        print(f"{m:<20} | {acc_means[i]:.2f} ± {acc_stds[i]:.2f}   | {rec_means[i]:.2f} ± {rec_stds[i]:.2f}")


if __name__ == "__main__":
    plot_final_comparison()