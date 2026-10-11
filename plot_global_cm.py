import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置 =================
LOG_FILE = "log_single_frame_scratch.txt"
SAVE_NAME = "global_confusion_matrix5.png"
LABELS = ["Daily Activity", "Fall"]  # 类别名称
FONT_FAMILY = "Times New Roman"


# =======================================

def parse_confusion_matrices(file_path):
    """提取所有Fold的混淆矩阵并求和"""
    # 初始化 2x2 矩阵
    total_cm = np.zeros((2, 2), dtype=int)

    # 匹配模式：寻找 [[nums] [nums]] 格式
    # 日志示例:
    # [[4079   72]
    #  [  38  391]]
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 使用正则找到所有矩阵块
    # 匹配 [[A B]\n [C D]]
    pattern = re.compile(r'\[\[\s*(\d+)\s+(\d+)\]\s*\n\s*\[\s*(\d+)\s+(\d+)\]\]')
    matches = pattern.findall(content)

    print(f"找到 {len(matches)} 个混淆矩阵，正在合并...")

    for m in matches:
        # 将字符串转为整数并累加
        matrix = np.array([
            [int(m[0]), int(m[1])],
            [int(m[2]), int(m[3])]
        ])
        total_cm += matrix

    return total_cm


def plot_confusion_matrix(cm):
    if np.sum(cm) == 0:
        print("未找到混淆矩阵数据。")
        return

    # 计算百分比矩阵
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # 设置样式
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['font.size'] = 14

    plt.figure(figsize=(7, 6))

    # 使用 Seaborn 绘制热力图
    # annot 格式：显示 "数量\n(百分比%)"
    annot_labels = (np.asarray(["{0}\n({1:.1f}%)".format(val, pct)
                                for val, pct in zip(cm.flatten(), cm_percent.flatten())])
                    ).reshape(2, 2)

    sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues',
                xticklabels=LABELS, yticklabels=LABELS,
                cbar=False, square=True, linewidths=1, linecolor='black',
                annot_kws={"fontsize": 16, "fontweight": "bold"})

    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.title('Global Confusion Matrix (LOSO 9 Folds)', fontsize=16, pad=15)

    plt.tight_layout()
    plt.savefig(SAVE_NAME, dpi=300)
    print(f"图像已保存至: {SAVE_NAME}")
    plt.show()


if __name__ == "__main__":
    cm = parse_confusion_matrices(LOG_FILE)
    plot_confusion_matrix(cm)