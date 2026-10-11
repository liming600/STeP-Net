import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ================= 配置区域 =================
NUM_SUBJECTS = 9
FONT_FAMILY = 'serif'
SAVE_NAME = 'loso_cv_scheme_ieee_gray_safe_v2.png'

# ================= IEEE + 灰度安全配色 =================
COLOR_TRAIN = '#FFFFFF'   # Train：白色
COLOR_VAL   = '#7F7F7F'   # Validation：中灰
COLOR_TEST  = '#1F4E79'   # Test：IEEE 深蓝（灰度下近黑）

EDGE_COLOR = 'black'


def plot_loso_scheme():
    # -------- 全局字体（进一步放大）--------
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['font.size'] = 26
    plt.rcParams['font.weight'] = 'bold'

    # -------- 画布（加高，给 Legend 留空间）--------
    fig, ax = plt.subplots(figsize=(18, 16))

    # -------- LOSO 网格 --------
    for fold in range(NUM_SUBJECTS):
        test_idx = fold
        val_idx = (fold + 1) % NUM_SUBJECTS

        for subj in range(NUM_SUBJECTS):
            if subj == test_idx:
                color = COLOR_TEST
            elif subj == val_idx:
                color = COLOR_VAL
            else:
                color = COLOR_TRAIN

            rect = mpatches.Rectangle(
                (subj, NUM_SUBJECTS - 1 - fold),
                1, 1,
                linewidth=2.2,
                edgecolor=EDGE_COLOR,
                facecolor=color
            )
            ax.add_patch(rect)

    # -------- 坐标轴 --------
    ax.set_xlim(0, NUM_SUBJECTS)
    ax.set_ylim(0, NUM_SUBJECTS)

    ax.set_xticks(np.arange(NUM_SUBJECTS) + 0.5)
    ax.set_xticklabels(
        [f"S{i+1}" for i in range(NUM_SUBJECTS)],
        fontsize=26, fontweight='bold'
    )

    ax.set_yticks(np.arange(NUM_SUBJECTS) + 0.5)
    ax.set_yticklabels(
        [f"Fold {NUM_SUBJECTS - i}" for i in range(NUM_SUBJECTS)],
        fontsize=26, fontweight='bold'
    )

    ax.set_xlabel(
        'Subjects (S1–S9)',
        fontsize=30, fontweight='bold', labelpad=22
    )
    ax.set_ylabel(
        'Cross-Validation Iterations',
        fontsize=30, fontweight='bold', labelpad=22
    )

    # 移除多余的刻度线和边框
    ax.tick_params(axis='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # -------- Legend（关键修改点）--------
    legend_patches = [
        mpatches.Patch(
            facecolor=COLOR_TRAIN, edgecolor=EDGE_COLOR,
            linewidth=1.8, label='Training (7 Subjects)'
        ),
        mpatches.Patch(
            facecolor=COLOR_VAL, edgecolor=EDGE_COLOR,
            linewidth=1.8, label='Validation (1 Subject)'
        ),
        mpatches.Patch(
            facecolor=COLOR_TEST, edgecolor=EDGE_COLOR,
            linewidth=1.8, label='Testing (1 Subject)'
        )
    ]

    ax.legend(
        handles=legend_patches,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.18),   # ⬅️ 再上移
        ncol=3,
        frameon=False,
        fontsize=26,
        prop={'weight': 'bold', 'size': 26},
        handlelength=2.2,
        columnspacing=2.0
    )

    # -------- 标题 --------
    plt.title(
        'Strict Leave-One-Subject-Out Scheme',
        fontsize=34, fontweight='bold', y=1.23
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])  # ⬅️ 明确给顶部留白
    plt.savefig(SAVE_NAME, dpi=300)
    print(f"IEEE 灰度安全 LOSO 示意图已保存为: {SAVE_NAME}")
    # plt.show()


if __name__ == "__main__":
    plot_loso_scheme()
