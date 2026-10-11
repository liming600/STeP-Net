# cloud_inference_api.py (在云端服务器上运行，升级了采样逻辑)

from flask import Flask, request, jsonify
import numpy as np
import torch
from collections import deque
import sys
import os
import open3d as o3d # [核心修改] 引入 Open3D

# --- 核心：挂载训练项目的路径 ---
sys.path.append('/workspace/ZLL/point-MAE')
try:
    from utils.config import get_config
    from tools import builder
except ImportError as e:
    sys.exit(f"Error importing point-MAE modules: {e}")

app = Flask(__name__)

# --- 全局变量 ---
MODEL = None
DEVICE = None
FRAME_BUFFER = deque(maxlen=16)
N_POINTS = 1024
SCALE_FACTOR = 3.0


def load_model():
    """在 Flask 启动时加载模型"""
    global MODEL, DEVICE

    # !!! 请务必将这里的路径修改为你最终选定的模型 !!!
    config_path = '/workspace/ZLL/point-MAE/cfgs/finetune_sequence_lstm.yaml'
    ckpt_path = '/workspace/ZLL/point-MAE/experiments/finetune_sequence_lstm/cfgs/loso_lstm_finetune/fold_test_subj_4/ckpt-best.pth'

    class Args:
        config = config_path
        opts = []

    config = get_config(Args(), None)
    MODEL = builder.model_builder(config.model)
    builder.load_model(MODEL, ckpt_path)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL.to(DEVICE)
    MODEL.eval()
    print(f"--- Model loaded successfully on {DEVICE} ---")


def unify_point_count(points_np, target_count):
    """
    [核心修改]
    将点云的点数统一到目标数量。
    - 使用 FPS 进行下采样。
    - 使用随机重复进行上采样。
    """
    current_count = points_np.shape[0]

    if current_count == 0:
        return np.zeros((target_count, 3), dtype=np.float32)

    if current_count > target_count:
        # --- 使用 Open3D 进行 FPS 下采样 ---
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        pcd_downsampled = pcd.farthest_point_down_sample(target_count)
        return np.asarray(pcd_downsampled.points)

    elif current_count < target_count:
        # --- 使用 NumPy 进行随机重复上采样 ---
        points_to_add = target_count - current_count
        additional_indices = np.random.choice(current_count, size=points_to_add, replace=True)
        additional_points = points_np[additional_indices, :]
        return np.vstack((points_np, additional_points))

    else:
        return points_np


@app.route('/api/infer', methods=['POST'])
def infer():
    """接收点云数据，进行推理并返回结果"""
    global FRAME_BUFFER

    data = request.get_json()
    if not data or 'points' not in data:
        return jsonify({'error': 'Missing points data'}), 400

    points = np.array(data['points'], dtype=np.float32)

    # 2. 归一化 (保持不变)
    center = np.median(points, axis=0)
    points_normalized = (points - center) / SCALE_FACTOR

    # 3. 统一点数 (调用新函数)
    points_unified = unify_point_count(points_normalized, N_POINTS)

    # 4. 更新缓冲区并推理
    FRAME_BUFFER.append(torch.from_numpy(points_unified).float())

    prediction = 0 # 默认为正常
    if len(FRAME_BUFFER) == 16:
        sequence = torch.stack(list(FRAME_BUFFER), dim=0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = MODEL(sequence)
            prediction = logits.argmax(-1).item()

    # 5. 返回结果
    result = "FALL_DETECTED" if prediction == 1 else "NORMAL"
    return jsonify({'status': result})


if __name__ == '__main__':
    load_model()
    # 在生产环境中，建议使用 Gunicorn 或 uWSGI
    app.run(host='0.0.0.0', port=5000)