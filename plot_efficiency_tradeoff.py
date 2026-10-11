import matplotlib.pyplot as plt
import numpy as np

# ================= 1. Experimental Data =================
models = ['STeP-SF', 'STeP-TC', 'STeP-TL', 'STeP-TG', 'STeP-TTr', 'STeP-TG-Scratch']
latency = [4.16, 43.83, 44.50, 44.31, 44.15, 44.43]
# Accuracy (Placeholder for Scratch model, please update if you have real data)

accuracy = [98.74, 98.89, 98.91, 99.38, 98.97, 92.5]
trainable_params = [21.99, 0.41, 1.22, 0.92, 4.20, 22.91]
# Bubble sizes
sizes = [p * 180 + 400 for p in trainable_params]  # Slightly larger bubbles

# ================= 2. Plotting Configuration =================
plt.figure(figsize=(22, 16))  # Larger canvas
plt.rcParams['font.family'] = 'serif'
# We will set specific font sizes manually for elements, but set a base here
plt.rcParams['font.size'] = 24

# Scatter Plot
# zorder=10 ensures bubbles are on top of grid
scatter = plt.scatter(latency, accuracy, s=sizes, c=accuracy, cmap='viridis',
                      alpha=0.85, edgecolors='black', linewidth=2.5, zorder=10)

# Colorbar
cbar = plt.colorbar(scatter, pad=0.02)
cbar.set_label('Accuracy (%)', rotation=270, labelpad=50, fontsize=32, fontweight='bold')
cbar.ax.tick_params(labelsize=28)
for l in cbar.ax.yaxis.get_ticklabels():
    l.set_weight("bold")

# ================= 3. Smart Label Positioning =================
# Strategy:
# TG (Highest) -> Go Up
# TTr (Mid-High) -> Go Right
# TL (Mid-Low) -> Go Left
# TC (Lowest of cluster) -> Go Down
# SF -> Go Right (plenty of space)
# Scratch -> Go Right/Up (plenty of space)

offsets = {
    'STeP-SF': (-40, -80),  # Right
    'STeP-TG': (0, 130),  # Up (High)
    'STeP-TTr': (160, -20),  # Right
    'STeP-TL': (-160, 80),  # Left
    'STeP-TC': (20, -150),  # Down-Left
    'STeP-TG-Scratch': (100, 80)  # Top-Right
}

alignments = {
    'STeP-SF': ('left', 'center'),
    'STeP-TG': ('center', 'bottom'),
    'STeP-TTr': ('left', 'center'),
    'STeP-TL': ('right', 'center'),
    'STeP-TC': ('right', 'top'),
    'STeP-TG-Scratch': ('left', 'bottom')
}

# Annotation Loop
for i, name in enumerate(models):
    label_text = f"{name}\n({trainable_params[i]:.2f}M)"

    dx, dy = offsets.get(name, (10, 10))
    ha, va = alignments.get(name, ('center', 'center'))

    plt.annotate(label_text,
                 (latency[i], accuracy[i]),
                 xytext=(dx, dy),
                 textcoords='offset points',
                 fontsize=26, fontweight='bold',  # Increased annotation font
                 ha=ha, va=va,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2",
                                 color='#444444', linewidth=4),  # Thicker arrows
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.95, ec="#888888", lw=3))

# ================= 4. Axis and Decoration =================
plt.title('Trade-off: Accuracy vs. Efficiency (Trainable Params)', fontsize=40, pad=60, fontweight='bold')
plt.xlabel('Inference Latency (ms) [Lower is Better]', fontsize=32, fontweight='bold')
plt.ylabel('Accuracy (%) [Higher is Better]', fontsize=32, fontweight='bold')

# Bold and large ticks (Size 30 as requested)
plt.xticks(fontsize=30, fontweight='bold')
plt.yticks(fontsize=30, fontweight='bold')

# Expanded Limits to prevent clipping
# X: -20 to 180 to accomodate labels
# Y: 91 to 102 to accomodate labels
plt.xlim(-20, 200)
plt.ylim(91, 102.5)

plt.grid(True, linestyle='--', alpha=0.4, linewidth=3)
plt.tight_layout()

save_name = 'final_efficiency_tradeoff_v5_large.png'
plt.savefig(save_name, dpi=300)
print(f"Plot saved to: {save_name}")
plt.show()