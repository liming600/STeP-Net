import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np


def parse_log_file(file_path):
    """解析日志文件，提取每个 Fold 的训练数据"""
    data = {}
    current_fold = None

    fold_pattern = re.compile(r'LOSO FOLD (\d+)/\d+')
    train_pattern = re.compile(
        r'\[Training\] EPOCH:\s+(\d+).*Losses = \[\'([\d\.]+)\'.*lr = ([\d\.e-]+)'
    )
    val_pattern = re.compile(
        r'\[Validation\] EPOCH:\s+(\d+)\s+acc = ([\d\.]+)'
    )

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            fold_match = fold_pattern.search(line)
            if fold_match:
                current_fold = int(fold_match.group(1))
                data[current_fold] = {
                    'train_epoch': [], 'loss': [], 'lr': [],
                    'val_epoch': [], 'val_acc': []
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


def get_local_epoch_range(data):
    """获取单个数据字典的 epoch 范围"""
    all_train_epochs = []
    all_val_epochs = []
    for fold_data in data.values():
        all_train_epochs.extend(fold_data.get('train_epoch', []))
        all_val_epochs.extend(fold_data.get('val_epoch', []))

    all_epochs = all_train_epochs + all_val_epochs
    if all_epochs:
        return min(all_epochs), max(all_epochs)
    return 0, 100


def set_ieee_style():
    """设置 IEEE 论文风格"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'font.weight': 'bold',
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.2
    })


def plot_training_dynamics(data, save_name):
    """
    绘制训练动态图：左图训练 Loss，右图验证准确率

    Args:
        data: 解析后的数据字典
        save_name: 保存文件名
    """
    set_ieee_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # 颜色和标记
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']

    # 获取当前数据的 epoch 范围
    min_epoch, max_epoch = get_local_epoch_range(data)
    xlim_range = (min_epoch, max_epoch + 1)

    # ========== 左图：Training Loss ==========
    for i, (fold_id, fold_data) in enumerate(sorted(data.items())):
        epochs = fold_data.get('train_epoch', [])
        losses = fold_data.get('loss', [])

        if epochs and losses:
            ax1.plot(
                epochs, losses,
                color=colors[(fold_id - 1) % len(colors)],
                label=f'F{fold_id}',
                marker=markers[(fold_id - 1) % len(markers)],
                markersize=3.5,
                markerfacecolor='white',
                markeredgewidth=0.8,
                linewidth=1.2
            )

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_xlim(xlim_range)
    ax1.set_ylim(bottom=0)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 添加子图标签 (a) - 位于纵轴标签上方，与纵轴最高位置平齐
    ax1.text(-0.12, 1.02, '(a)', transform=ax1.transAxes,
             fontsize=8, fontweight='bold', va='bottom')

    # ========== 右图：Validation Accuracy ==========
    for i, (fold_id, fold_data) in enumerate(sorted(data.items())):
        epochs = fold_data.get('val_epoch', [])
        accuracies = fold_data.get('val_acc', [])

        if epochs and accuracies:
            ax2.plot(
                epochs, accuracies,
                color=colors[(fold_id - 1) % len(colors)],
                label=f'F{fold_id}',
                marker=markers[(fold_id - 1) % len(markers)],
                markersize=3.5,
                markerfacecolor='white',
                markeredgewidth=0.8,
                linewidth=1.2
            )

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_xlim(xlim_range)
    ax2.set_ylim(80, 101)
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 添加子图标签 (b) - 位于纵轴标签上方，与纵轴最高位置平齐
    ax2.text(-0.12, 1.02, '(b)', transform=ax2.transAxes,
             fontsize=8, fontweight='bold', va='bottom')

    # ========== 图例：铺平在一行，放在顶部中央 ==========
    handles, labels = ax1.get_legend_handles_labels()

    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(handles),
        frameon=True,
        facecolor='white',
        framealpha=0.9,
        edgecolor='none',
        fontsize=7,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=0.8
    )

    # 调整子图位置，为顶部图例留出空间
    plt.subplots_adjust(top=0.88, left=0.1, right=0.95, wspace=0.3)

    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"已保存: {save_name}")

    return fig, (ax1, ax2)


def plot_training_dynamics_with_mean(data, save_name):
    """
    绘制训练动态图，额外添加均值曲线和标准差阴影

    Args:
        data: 解析后的数据字典
        save_name: 保存文件名
    """
    set_ieee_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    colors = ['#1f77b4', '#ff7f0e']

    # 获取当前数据的 epoch 范围
    min_epoch, max_epoch = get_local_epoch_range(data)
    xlim_range = (min_epoch, max_epoch + 1)

    # ========== 计算均值和标准差 ==========
    def compute_stats(data, metric_type, epoch_type):
        """计算跨所有fold的均值和标准差"""
        all_epochs_list = []
        all_values_list = []

        for fold_id, fold_data in data.items():
            if epoch_type == 'train':
                epochs = fold_data.get('train_epoch', [])
                values = fold_data.get(metric_type, [])
            else:
                epochs = fold_data.get('val_epoch', [])
                values = fold_data.get(metric_type, [])

            all_epochs_list.append(epochs)
            all_values_list.append(values)

        if not all_values_list:
            return [], [], []

        # 找到最大 epoch 数
        max_len = max([len(v) for v in all_values_list])

        # 对齐所有 fold 的 epoch
        aligned_values = []
        aligned_epochs = []

        for epochs, values in zip(all_epochs_list, all_values_list):
            epoch_value_map = dict(zip(epochs, values))
            aligned_values.append([epoch_value_map.get(ep, np.nan) for ep in range(max_len)])
            aligned_epochs.append(list(range(max_len)))

        # 计算均值和标准差
        mean_values = np.nanmean(aligned_values, axis=0)
        std_values = np.nanstd(aligned_values, axis=0)

        return aligned_epochs[0] if aligned_epochs else [], mean_values, std_values

    # 计算 Loss 的统计量
    loss_epochs, loss_mean, loss_std = compute_stats(data, 'loss', 'train')
    # 计算 Accuracy 的统计量
    acc_epochs, acc_mean, acc_std = compute_stats(data, 'val_acc', 'val')

    # ========== 左图：Training Loss with mean ± std ==========
    if len(loss_epochs) > 0:
        ax1.plot(loss_epochs, loss_mean, color=colors[0], linewidth=2, label='Mean')
        ax1.fill_between(loss_epochs, loss_mean - loss_std, loss_mean + loss_std,
                         color=colors[0], alpha=0.2, label='±1 Std')

        # 绘制原始 folds 的淡色线条
        for fold_id, fold_data in sorted(data.items()):
            epochs = fold_data.get('train_epoch', [])
            losses = fold_data.get('loss', [])
            if epochs and losses:
                ax1.plot(epochs, losses, color='gray', alpha=0.3, linewidth=0.8)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_xlim(xlim_range)
    ax1.set_ylim(bottom=0)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(loc='upper right', fontsize=7, frameon=True, facecolor='white', framealpha=0.8)

    # 添加子图标签 (a) - 位于纵轴标签上方，与纵轴最高位置平齐
    ax1.text(-0.12, 1.02, '(a)', transform=ax1.transAxes,
             fontsize=8, fontweight='bold', va='bottom')

    # ========== 右图：Validation Accuracy with mean ± std ==========
    if len(acc_epochs) > 0:
        ax2.plot(acc_epochs, acc_mean, color=colors[0], linewidth=2, label='Mean')
        ax2.fill_between(acc_epochs, acc_mean - acc_std, acc_mean + acc_std,
                         color=colors[0], alpha=0.2, label='±1 Std')

        # 绘制原始 folds 的淡色线条
        for fold_id, fold_data in sorted(data.items()):
            epochs = fold_data.get('val_epoch', [])
            accuracies = fold_data.get('val_acc', [])
            if epochs and accuracies:
                ax2.plot(epochs, accuracies, color='gray', alpha=0.3, linewidth=0.8)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_xlim(xlim_range)
    ax2.set_ylim(80, 101)
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='lower right', fontsize=7, frameon=True, facecolor='white', framealpha=0.8)

    # 添加子图标签 (b) - 位于纵轴标签上方，与纵轴最高位置平齐
    ax2.text(-0.12, 1.02, '(b)', transform=ax2.transAxes,
             fontsize=8, fontweight='bold', va='bottom')

    # 调整子图位置
    plt.subplots_adjust(left=0.1, right=0.95, wspace=0.3, top=0.92)

    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"已保存: {save_name}")

    return fig, (ax1, ax2)


if __name__ == "__main__":
    # 解析日志文件
    log_overfit = "log_overfit.txt"
    log_gru = "log_gru.txt"

    data_overfit = parse_log_file(log_overfit)
    data_gru = parse_log_file(log_gru)

    # 打印各自 epoch 范围
    overfit_min, overfit_max = get_local_epoch_range(data_overfit)
    gru_min, gru_max = get_local_epoch_range(data_gru)
    print(f"Overfit Epoch 范围: {overfit_min} - {overfit_max}")
    print(f"GRU Epoch 范围: {gru_min} - {gru_max}")

    # ========== 生成 Overfit 实验的训练动态图 ==========
    print("\n=== 生成 Overfit 实验训练动态图 ===")
    plot_training_dynamics(
        data_overfit,
        save_name='overfit_training_dynamics.png'
    )

    # ========== 生成 GRU 实验的训练动态图 ==========
    print("\n=== 生成 GRU 实验训练动态图 ===")
    plot_training_dynamics(
        data_gru,
        save_name='gru_training_dynamics.png'
    )

    # ========== 生成带均值曲线的版本 ==========
    print("\n=== 生成带均值曲线的版本 ===")
    plot_training_dynamics_with_mean(
        data_overfit,
        save_name='overfit_training_dynamics_with_mean.png'
    )

    plot_training_dynamics_with_mean(
        data_gru,
        save_name='gru_training_dynamics_with_mean.png'
    )

    print("\n所有图表生成完成！")