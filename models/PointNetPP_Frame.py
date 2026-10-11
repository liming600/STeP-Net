import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import MODELS
from pointnet2_ops.pointnet2_modules import PointnetSAModuleMSG, PointnetSAModule
from utils.logger import print_log


@MODELS.register_module()
class PointNetPP_Frame(nn.Module):
    """PointNet++ 帧级分类器。

    遵循 PointNet++ 标准分类架构 (Qi et al., 2017)，
    与 "Fall Detection using LiDAR Sensor with PointNet++" (Ayyagari et al., 2025) 一致。

    结构: 2× SA-MSG → 1× SA-Global → 2× FC
    对输入序列每帧独立分类，取最后一帧输出。
    """
    def __init__(self, config):
        super().__init__()
        self.logger_name = 'PointNetPP_Frame'
        print_log(f'[{self.logger_name}] Building model...', logger=self.logger_name)

        self.config = config

        self.SA_modules = nn.ModuleList()
        sa_configs = config.sa_modules

        import copy as _copy
        for sa_cfg in sa_configs:
            npoint = sa_cfg.get('npoint', None)
            radii = sa_cfg.get('radii', None)
            nsamples = sa_cfg.get('nsamples', None)
            mlps = _copy.deepcopy(sa_cfg.get('mlps', sa_cfg.get('mlp', None)))
            if mlps is None:
                continue
            if not isinstance(mlps[0], (list, tuple)):
                mlps = [mlps]

            if npoint is not None and radii is not None and len(mlps) > 1:
                self.SA_modules.append(
                    PointnetSAModuleMSG(
                        npoint=npoint,
                        radii=radii,
                        nsamples=nsamples,
                        mlps=mlps,
                        use_xyz=True,
                    )
                )
            else:
                self.SA_modules.append(
                    PointnetSAModule(
                        npoint=npoint,
                        radius=sa_cfg.get('radius', None),
                        nsample=sa_cfg.get('nsample', None),
                        mlp=mlps[0] if isinstance(mlps[0], list) else mlps,
                        use_xyz=True,
                    )
                )

        classifier_in_dim = config.get('classifier_in_dim',
                                       config.get('frame_feature_dim', 256))
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, config.num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_sa(self, points):
        xyz = points
        features = points.transpose(1, 2).contiguous()
        for sa_module in self.SA_modules:
            xyz, features = sa_module(xyz, features)
        return features.squeeze(-1)

    def get_loss_acc(self, ret, gt):
        # 单帧分类：ret: (B, C), gt: (B,)
        logits = ret
        gt = gt.long()
        loss = F.cross_entropy(logits, gt)
        pred = logits.argmax(dim=1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def forward(self, point_cloud):
        # 单帧输入 (B, N, 3) → (B, C)，与原论文逐帧分类一致
        frame_features = self._forward_sa(point_cloud)
        logits = self.classifier(frame_features)
        return logits
