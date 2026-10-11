import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ================= 配置 =================
NUM_SUBJECTS = 9
SAVE_NAME = "loso_scheme.png"

# 配色（保持不变）
COLOR_TRAIN = "#FFFFFF"
COLOR_VAL   = "#7F7F7F"
COLOR_TEST  = "#1F4E79"

EDGE_COLOR = "black"

# ================= IEEE 单栏图规格 =================
FIG_WIDTH = 4.2
FIG_HEIGHT = 2.8     # ↓高度缩减
DPI = 300

FONT_SIZE_TICK = 8
FONT_SIZE_LABEL = 9
FONT_SIZE_LEGEND = 8


def plot_loso_scheme():

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = FONT_SIZE_LABEL
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.linewidth"] = 0.8

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # ================= LOSO 网格 =================
    for fold in range(NUM_SUBJECTS):

        test_idx = fold
        val_idx = (fold + 1) % NUM_SUBJECTS

        for subj in range(NUM_SUBJECTS):

            if subj == test_idx:
                color = COLOR_TEST
                hatch = "///"      # Testing

            elif subj == val_idx:
                color = COLOR_VAL
                hatch = "..."      # Validation

            else:
                color = COLOR_TRAIN
                hatch = None       # Training

            rect = mpatches.Rectangle(
                (subj, NUM_SUBJECTS - 1 - fold),
                1, 1,
                linewidth=0.6,
                edgecolor=EDGE_COLOR,
                facecolor=color,
                hatch=hatch
            )

            ax.add_patch(rect)

    # ================= 坐标轴 =================
    ax.set_xlim(0, NUM_SUBJECTS)
    ax.set_ylim(0, NUM_SUBJECTS)

    ax.set_xticks(np.arange(NUM_SUBJECTS) + 0.5)
    ax.set_xticklabels(
        [f"S{i+1}" for i in range(NUM_SUBJECTS)],
        fontsize=FONT_SIZE_TICK,
        fontweight="bold"
    )

    ax.set_yticks(np.arange(NUM_SUBJECTS) + 0.5)
    ax.set_yticklabels(
        [f"F{NUM_SUBJECTS - i}" for i in range(NUM_SUBJECTS)],
        fontsize=FONT_SIZE_TICK,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Subjects",
        fontsize=FONT_SIZE_LABEL,
        # fontweight="bold",
        labelpad=3
    )

    ax.set_ylabel(
        "CV Iterations",
        fontsize=FONT_SIZE_LABEL,
        # fontweight="bold",
        labelpad=3
    )

    # ================= 网格比例 3:2 =================
    # 取消正方形比例，设置为长方形
    ax.set_aspect(2/3)

    ax.tick_params(axis="both", length=0, pad=1.5)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # ================= Legend =================
    legend_patches = [

        mpatches.Patch(
            facecolor=COLOR_TRAIN,
            edgecolor=EDGE_COLOR,
            linewidth=0.6,
            label="Training"
        ),

        mpatches.Patch(
            facecolor=COLOR_VAL,
            edgecolor=EDGE_COLOR,
            linewidth=0.6,
            hatch="...",
            label="Validation"
        ),

        mpatches.Patch(
            facecolor=COLOR_TEST,
            edgecolor=EDGE_COLOR,
            linewidth=0.6,
            hatch="///",
            label="Testing"
        )
    ]

    ax.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
        prop={"weight": "bold", "size": FONT_SIZE_LEGEND},
        handlelength=1.4,
        columnspacing=0.8,
        handletextpad=0.4
    )

    # ================= 保存 =================
    plt.tight_layout(pad=0.3)

    plt.savefig(
        SAVE_NAME,
        dpi=DPI,
        bbox_inches="tight"
    )

    print("Figure saved:", SAVE_NAME)


if __name__ == "__main__":
    plot_loso_scheme()