import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def parse_log_file(file_path):
    """解析日志文件，提取每个 Fold 的训练与验证数据"""
    data = {}
    current_fold = None

    fold_pattern = re.compile(r'LOSO FOLD (\d+)/\d+')
    train_pattern = re.compile(
        r"\[Training\] EPOCH:\s+(\d+).*Losses = \['([\d\.]+)'.*lr = ([\d\.e-]+)"
    )
    val_pattern = re.compile(
        r"\[Validation\] EPOCH:\s+(\d+)\s+acc = ([\d\.]+)"
    )

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            fold_match = fold_pattern.search(line)
            if fold_match:
                current_fold = int(fold_match.group(1))
                data[current_fold] = {
                    'train_epoch': [],
                    'loss': [],
                    'lr': [],
                    'val_epoch': [],
                    'val_acc': []
                }
                continue

            if current_fold is None:
                continue

            train_match = train_pattern.search(line)
            if train_match:
                data[current_fold]['train_epoch'].append(int(train_match.group(1)))
                data[current_fold]['loss'].append(float(train_match.group(2)))
                data[current_fold]['lr'].append(float(train_match.group(3)))

            val_match = val_pattern.search(line)
            if val_match:
                data[current_fold]['val_epoch'].append(int(val_match.group(1)))
                data[current_fold]['val_acc'].append(float(val_match.group(2)))

    return data


# =================== 统一 IEEE 绘图风格 ===================
def set_ieee_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.5
    })


def style_ticks(ax):
    for label in ax.get_xticklabels():
        label.set_fontsize(8)
        label.set_fontweight('bold')

    for label in ax.get_yticklabels():
        label.set_fontsize(8)
        label.set_fontweight('bold')


def plot_single_metric(x, ys, ylabel, save_name, ylim=None, linestyle='-'):
    """单指标绘图函数（保留备用）"""
    fig, ax = plt.subplots(figsize=(4.2, 2.8))

    style_ticks(ax)

    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']

    for fold_id, y in ys.items():
        ax.plot(
            x[fold_id],
            y,
            color=colors[(fold_id - 1) % len(colors)],
            label=f'Fold {fold_id}',
            linestyle=linestyle,
            marker=markers[(fold_id - 1) % len(markers)],
            markersize=4,
            markerfacecolor='white',
            markeredgewidth=0.9
        )

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)

    if ylim:
        ax.set_ylim(*ylim)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)

    ax.legend(
        loc='upper right',
        frameon=True,
        facecolor='white',
        framealpha=0.8,
        edgecolor='none'
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"已保存: {save_name}")


def plot_loss_and_acc_combined(data, save_name):
    """
    在同一画布中展示：
    (a) Training Loss per Fold
    (b) Validation Accuracy per Fold
    图例放在两幅子图的最上侧中央
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))

    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']

    # 用于存储图例的句柄和标签
    legend_handles = []
    legend_labels = []

    # =============== (a) Training Loss ===============
    ax = axes[0]
    for fold_id, fold_data in data.items():
        line, = ax.plot(
            fold_data['train_epoch'],
            fold_data['loss'],
            color=colors[(fold_id - 1) % len(colors)],
            linestyle='-',
            marker=markers[(fold_id - 1) % len(markers)],
            markersize=3.5,
            markerfacecolor='white',
            markeredgewidth=0.8,
            label=f'F{fold_id}'
        )
        legend_handles.append(line)
        legend_labels.append(f'F{fold_id}')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    style_ticks(ax)
    ax.text(
        -0.2, 1.05, '(a)',
        transform=ax.transAxes,
        fontsize=9,
        fontweight='bold',
        va='top',
        ha='left'
    )

    # =============== (b) Validation Accuracy ===============
    ax = axes[1]
    for fold_id, fold_data in data.items():
        line, = ax.plot(
            fold_data['val_epoch'],
            fold_data['val_acc'],
            color=colors[(fold_id - 1) % len(colors)],
            linestyle='-',
            marker=markers[(fold_id - 1) % len(markers)],
            markersize=3.5,
            markerfacecolor='white',
            markeredgewidth=0.8,
            label=f'F{fold_id}'
        )
        # 注意：这里不需要重复添加图例句柄，因为上面已经添加了

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_ylim(80, 100.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    style_ticks(ax)
    ax.text(
        -0.26, 1.05, '(b)',
        transform=ax.transAxes,
        fontsize=9,
        fontweight='bold',
        va='top',
        ha='left'
    )

    # 去除重复的图例句柄和标签（保持唯一）
    unique_handles = []
    unique_labels = []
    for handle, label in zip(legend_handles, legend_labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    # 图例放在两幅子图的最上侧中央
    fig.legend(
        unique_handles, unique_labels,
        loc='upper center',  # 位置：上侧中央
        bbox_to_anchor=(0.5, 1.08),  # 锚点位置：x=0.5(水平居中), y=1.08(图上方)
        frameon=True,
        facecolor='white',
        framealpha=0.8,
        edgecolor='none',
        ncol=len(unique_labels),  # 水平排列所有图例项
        prop={'weight': 'bold'}
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # 调整顶部空间，为图例留出位置
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"已保存: {save_name}")


if __name__ == "__main__":
    log_file = "log_overfit.txt"
    data = parse_log_file(log_file)

    set_ieee_style()

    # 组合图：Training Loss + Validation Accuracy
    plot_loss_and_acc_combined(
        data=data,
        save_name='TS_combined.png'
    )

    # 单独保留学习率图（可选）
    plot_single_metric(
        x={k: v['train_epoch'] for k, v in data.items()},
        ys={k: v['lr'] for k, v in data.items()},
        ylabel='Learning Rate',
        save_name='learning_rate.png',
        linestyle='--'
    )