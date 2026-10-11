import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- 全局绘图配置 ---
# 使用Seaborn的默认主题和字体设置，不再需要处理中文字体
sns.set_theme(style="whitegrid", font_scale=1.2)


def plot_confusion_matrix(cm, class_names, title, output_filename):
    """
    绘制并保存一个带标注的热力图格式的混淆矩阵。
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                linewidths=.5, cbar=False)

    # --- [修改] ---
    # 将所有标签和标题改为英文
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_title(title, fontsize=16, pad=20, weight='bold')

    plt.tight_layout()

    output_dir = 'analysis_plots'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, output_filename)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix plot saved to: {save_path}")
    plt.close(fig)


def calculate_metrics(cm):
    """从混淆矩阵计算'跌倒'类别的 Precision, Recall, F1-Score"""
    cm = np.array(cm)
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def plot_summary_chart(results, output_filename):
    """
    绘制一个分组柱状图来汇总所有模型的关键评估指标。
    """
    model_names = [name.replace('_', ' ') for name in results.keys()]
    precisions = [res['precision'] * 100 for res in results.values()]
    recalls = [res['recall'] * 100 for res in results.values()]
    f1_scores = [res['f1_score'] * 100 for res in results.values()]

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(18, 10))

    # --- [修改] ---
    # 将图例标签改为英文
    rects1 = ax.bar(x - width, precisions, width, label='Precision', color='cornflowerblue')
    rects2 = ax.bar(x, recalls, width, label='Recall', color='sandybrown')
    rects3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='mediumseagreen')

    # --- [修改] ---
    # 将Y轴标签和图表标题改为英文
    ax.set_ylabel('Score (%)', fontsize=14)
    ax.set_title('Performance Comparison of Models for "Fall" Detection', fontsize=18, pad=20, weight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=12)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(axis='x')
    ax.set_ylim(80, 102)

    ax.bar_label(rects1, padding=3, fmt='%.2f%%')
    ax.bar_label(rects2, padding=3, fmt='%.2f%%')
    ax.bar_label(rects3, padding=3, fmt='%.2f%%')

    fig.tight_layout()

    output_dir = 'analysis_plots'
    save_path = os.path.join(output_dir, output_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSummary chart saved to: {save_path}")
    plt.close(fig)


# ==============================================================================
# ---                       主执行区域：填入您的数据                     ---
# ==============================================================================
if __name__ == '__main__':

    # --- [修改] ---
    # 将类别名称改为英文
    class_names = ['Daily Activity', 'Fall']

    # --- 在这里填入您所有实验的最终【混淆矩阵】 ---
    # 格式: [[TN, FP], [FN, TP]]
    cm_from_scratch = [[5477, 154], [62, 680]]
    cm_single_frame = [[5554, 77], [46, 696]]  # 请根据您的日志确认此数据

    # --- 时序模型 (使用“序列一致增强”的结果) ---
    cm_lstm = [[336, 0], [2, 38]]
    cm_gru = [[336, 0], [5, 35]]
    cm_transformer = [[336, 0], [0, 40]]
    cm_1dcnn = [[336, 0], [6, 34]]  # 请填入1DCNN使用序列一致增强的最终测试结果

    # --- [修改] ---
    # 将标题改为英文
    all_cms = {
        'From_Scratch': (cm_from_scratch, "CM: From Scratch (Single-Frame)"),
        'Single_Frame': (cm_single_frame, "CM: SSL+FSL (Single-Frame)"),
        'LSTM_Sequential': (cm_lstm, "CM: Temporal - LSTM"),
        'GRU_Sequential': (cm_gru, "CM: Temporal - GRU"),
        'Transformer_Sequential': (cm_transformer, "CM: Temporal - Transformer"),
        '1DCNN_Sequential': (cm_1dcnn, "CM: Temporal - 1D-CNN")
    }

    # --- 1. 循环绘制所有混淆矩阵 ---
    print("--- Generating confusion matrix plots for all models ---")
    for model_key, (cm, title) in all_cms.items():
        plot_confusion_matrix(np.array(cm), class_names, title, f"cm_{model_key}.png")

    # --- 2. 循环计算所有指标并准备绘图数据 ---
    plot_data = {
        name: {'precision': p, 'recall': r, 'f1_score': f}
        for name, (p, r, f) in {
            key: calculate_metrics(cm_tuple[0]) for key, cm_tuple in all_cms.items()
        }.items()
    }

    # --- 3. 绘制最终的性能汇总图 ---
    print("\n--- Generating performance summary comparison chart ---")
    plot_summary_chart(plot_data, "model_performance_comparison.png")

    print("\n--- All plotting tasks completed! ---")