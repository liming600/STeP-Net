import torch
import os
from tools import builder
import numpy as np
from utils.logger import print_log, get_logger

try:
    import open3d as o3d
except ImportError:
    print("无法导入 open3d。请在您的环境中安装它: pip install open3d")
    o3d = None


def save_point_cloud(points, filename, vis_path):
    if not o3d:
        print("Open3D 未安装，无法保存点云文件。")
        return
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    points_np = points.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    o3d.io.write_point_cloud(os.path.join(vis_path, filename), pcd)


def run_vis(args, config):
    logger = get_logger(args.log_name)
    print_log('Visualizer start ... ', logger=logger)

    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    if config.model.NAME != 'Point_MAE':
        raise TypeError(f"可视化需要Point_MAE模型，但配置文件中的模型是 {config.model.NAME}")
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    builder.load_model(base_model, args.ckpts, logger=logger)

    # 保持为eval模式，因为vis=True应该会覆盖返回行为
    base_model.eval()

    vis_path = os.path.join(args.experiment_path, 'visualization_results')
    print_log(f'可视化结果将保存到: {vis_path}', logger=logger)

    with torch.no_grad():
        for i, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            if i >= 5:
                break

            points = data[0].cuda()

            try:
                # --- [最后的修改] ---
                # 显式传递 vis=True 参数来请求可视化的输出
                reconstructed_points, visible_points, _ = base_model(points, vis=True)
            except Exception as e:
                print_log(f"模型前向传播失败: {e}", logger=logger)
                print_log("请检查 Point_MAE.py 的 forward 函数定义。", logger=logger)
                return

            gt_points = points[0]
            recon_points = reconstructed_points[0]
            vis_points = visible_points[0]

            save_point_cloud(gt_points, f'sample_{i}_1_ground_truth.ply', vis_path)
            save_point_cloud(vis_points, f'sample_{i}_2_visible.ply', vis_path)
            save_point_cloud(recon_points, f'sample_{i}_3_reconstructed.ply', vis_path)

            print_log(f'已保存第 {i + 1} 个样本的可vis化结果。', logger=logger)

    print_log('可视化完成！', logger=logger)