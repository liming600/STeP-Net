import re
import matplotlib.pyplot as plt
import numpy as np

# ================= 配置区域 =================
LOG_FILE = "log_lstm.txt"
SAVE_NAME = "subject_consistency_boxplot1.png"
FONT_FAMILY = "Times New Roman"

# 颜色配置 (符合学术审美)
BOX_COLOR = "#DBE5F1"  # 浅蓝填充
EDGE_COLOR = "#366092"  # 深蓝边框
MEDIAN_COLOR = "#C0504D"  # 深红中位线
SCATTER_COLOR = "#7F7F7F"  # 灰色散点


# ===========================================

def parse_all_metrics(file_path):
    """
    从日志中提取每个 Subject 的所有评估指标
    返回: {subj_id: [acc, prec, recall, f1]}
    """
    data = {}

    # 匹配模式：Test metrics on Subject 1: {'accuracy': '97.60%', ...}
    # 我们需要提取这一行里的四个数值
    line_pattern = re.compile(r"Test metrics on Subject (\d+):")

    # 分别匹配各个指标 (兼容顺序变化)
    patterns = {
        'acc': re.compile(r"'accuracy': '([\d\.]+)%'"),
        'prec': re.compile(r"'precision': '([\d\.]+)%'"),  # weighted avg precision
        'rec': re.compile(r"'recall': '([\d\.]+)%'"),  # weighted avg recall
        'f1': re.compile(r"'f1-score': '([\d\.]+)%'")  # weighted avg f1
    }

    print(f"正在读取文件: {file_path} ...")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = line_pattern.search(line)
                if match:
                    subj_id = int(match.group(1))

                    # 在同一行中查找所有指标
                    metrics = []
                    try:
                        metrics.append(float(patterns['acc'].search(line).group(1)))
                        metrics.append(float(patterns['prec'].search(line).group(1)))
                        metrics.append(float(patterns['rec'].search(line).group(1)))
                        metrics.append(float(patterns['f1'].search(line).group(1)))

                        data[subj_id] = metrics
                    except AttributeError:
                        print(f"警告: Subject {subj_id} 的部分指标解析失败，已跳过。")
                        continue
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return None
    return data


def plot_consistency_boxplot(data):
    if not data:
        return

    subjects = sorted(data.keys())
    # 准备绘图数据：box_data 是一个列表的列表，每个子列表包含4个数值
    box_data = [data[s] for s in subjects]

    # 计算全局平均 Accuracy 用于画参考线
    all_accs = [d[0] for d in box_data]  # index 0 is accuracy
    mean_acc = np.mean(all_accs)

    # --- 设置论文标准样式 ---
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.5

    fig, ax = plt.subplots(figsize=(12, 6))

    # 1. 绘制箱线图
    # patch_artist=True 允许填充颜色
    # showfliers=False 不显示默认的离群点符号，我们后面自己画散点
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=False, widths=0.5,
                    medianprops=dict(color=MEDIAN_COLOR, linewidth=2),
                    boxprops=dict(facecolor=BOX_COLOR, color=EDGE_COLOR, linewidth=1.5),
                    whiskerprops=dict(color=EDGE_COLOR, linewidth=1.5),
                    capprops=dict(color=EDGE_COLOR, linewidth=1.5))

    # 2. 叠加散点 (Swarm Plot 效果)
    # 因为每组只有4个点，画出来能展示具体分布
    for i, metrics in enumerate(box_data):
        # x坐标：i+1 (因为boxplot从1开始)，加上一点点随机抖动
        x = np.random.normal(i + 1, 0.04, size=len(metrics))
        ax.scatter(x, metrics, alpha=0.7, color=SCATTER_COLOR, s=20, zorder=3)

    # 3. 绘制平均线
    ax.axhline(mean_acc, color='black', linestyle='--', linewidth=1, alpha=0.8,
               label=f'Avg Acc: {mean_acc:.2f}%', zorder=2)

    # 4. 美化与标注
    ax.set_xlabel('Test Subject ID', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance Metrics (%)', fontsize=14, fontweight='bold')
    ax.set_title('Consistency of Performance Metrics across Subjects (LOSO)', fontsize=16, pad=15)

    ax.set_xticklabels([f'S{s}' for s in subjects], fontsize=12)

    # Y轴范围自适应，保留一定边距
    flat_data = [item for sublist in box_data for item in sublist]
    min_val = min(flat_data)
    ax.set_ylim(bottom=max(80, min_val - 2), top=100.5)

    # 去除上方和右侧边框 (Spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 网格线
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # 添加图例 (手动创建一个Box对象作为图例)
    import matplotlib.patches as mpatches
    box_patch = mpatches.Patch(color=BOX_COLOR, label='Metric Distribution\n(Acc, Prec, Rec, F1)')
    scatter_line = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=SCATTER_COLOR, label='Individual Metric')
    mean_line = plt.Line2D([0], [0], color='black', linestyle='--', label='Mean Accuracy')

    ax.legend(handles=[box_patch, scatter_line, mean_line], loc='lower right', frameon=True)

    plt.tight_layout()

    plt.savefig(SAVE_NAME, dpi=300, bbox_inches='tight')
    print(f"图像已保存至: {SAVE_NAME}")
    plt.show()


# ================= 运行 =================
if __name__ == "__main__":
    results = parse_all_metrics(LOG_FILE)
    if results:
        plot_consistency_boxplot(results)