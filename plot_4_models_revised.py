import re
import matplotlib.pyplot as plt
import numpy as np

# ================= 配置区域 =================
LOG_FILES = {
    'TS': 'log_overfit.txt',
    'TF': 'loso_overfit_nofrozen.txt',
    'STeP': 'log_gru.txt'
}

SAVE_NAME = 'training_compare.png'

# ================= IEEE 图规格 =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['font.weight'] = 'bold'


# ================= 工具函数 =================

def parse_metrics(file_path):
    metrics = {}

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

            if "LOSO-CV FINAL RESULT" in content:

                acc_match = re.search(r'accuracy\s+(\d+\.\d+)\s+(\d+\.\d+)', content)
                if acc_match:
                    metrics['acc_mean'] = float(acc_match.group(1))
                    metrics['acc_std'] = float(acc_match.group(2))

                rec_match = re.search(r'fall_recall\s+(\d+\.\d+)\s+(\d+\.\d+)', content)
                if rec_match:
                    metrics['rec_mean'] = float(rec_match.group(1))
                    metrics['rec_std'] = float(rec_match.group(2))

    except Exception as e:
        print(f"Warning: 读取文件 {file_path} 失败: {e}")

    return metrics


def get_capped_errors(means, stds):

    lower_errors = stds
    upper_errors = np.minimum(stds, 100 - means)

    return np.array([lower_errors, upper_errors])


# ================= 主逻辑 =================

def plot_final_comparison():

    print("正在解析日志文件...")

    data = {}

    for model_name, path in LOG_FILES.items():
        res = parse_metrics(path)

        if not res:
            print(f"[Error] 无法解析 {path}")

        data[model_name] = res

    models = list(LOG_FILES.keys())

    acc_means = np.array([data[m].get('acc_mean', 0) for m in models])
    acc_stds = np.array([data[m].get('acc_std', 0) for m in models])

    rec_means = np.array([data[m].get('rec_mean', 0) for m in models])
    rec_stds = np.array([data[m].get('rec_std', 0) for m in models])

    acc_yerr = get_capped_errors(acc_means, acc_stds)
    rec_yerr = get_capped_errors(rec_means, rec_stds)

    # ================= 修改1：适度增大画布 =================
    fig, ax = plt.subplots(figsize=(4.2, 2.8))

    x = np.arange(len(models))

    # ================= 修改2：增大柱宽 =================
    width = 0.35

    rects1 = ax.bar(
        x - width / 2,
        acc_means,
        width,
        yerr=acc_yerr,
        label='Accuracy',
        color='#4F81BD',
        capsize=4,
        edgecolor='black',
        linewidth=0.6,
        hatch='//',
        error_kw={'elinewidth': 0.8, 'ecolor': 'black'}
    )

    rects2 = ax.bar(
        x + width / 2,
        rec_means,
        width,
        yerr=rec_yerr,
        label='Recall',
        color='#C0504D',
        capsize=4,
        edgecolor='black',
        linewidth=0.6,
        hatch='..',
        error_kw={'elinewidth': 0.8, 'ecolor': 'black'}
    )

    # ================= 数值标签 =================
    ax.bar_label(rects1, padding=3, fmt='%.1f', fontsize=8)
    ax.bar_label(rects2, padding=3, fmt='%.1f', fontsize=8)

    ax.set_ylabel('Performance (%)', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)

    ax.legend(
        loc='lower center',
        fontsize=8,
        frameon=True,
        edgecolor='black',
        ncol=2
    )

    # ================= 纵轴从65开始 =================
    ax.set_ylim(bottom=60, top=103)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        SAVE_NAME,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"✅ 绘图完成: {SAVE_NAME}")


if __name__ == "__main__":
    plot_final_comparison()