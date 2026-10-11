import torch
import torch.nn as nn
from .build import MODELS
from tools import builder
from utils.logger import print_log
# 遵循原项目风格，引入检查点工具
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message


@MODELS.register_module()
class PointMAE_Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logger_name = 'PointMAE_Transformer'
        print_log(f'[{self.logger_name}] Building model...', logger=self.logger_name)

        self.config = config

        # 1. 构建 Point-MAE 编码器 (PointTransformer)
        self.point_mae_encoder = builder.model_builder(config.point_mae_config)

        # 2. 冻结编码器参数
        self.freeze_encoder = config.get('freeze_encoder', True)
        if self.freeze_encoder:
            print_log(f'[{self.logger_name}] Freezing Point-MAE encoder parameters.', logger=self.logger_name)
            for param in self.point_mae_encoder.parameters():
                param.requires_grad = False

        # 3. 定义外部时序 Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.point_mae_config.trans_dim,
            nhead=config.temporal_transformer_config.num_heads,
            dim_feedforward=config.temporal_transformer_config.dim_feedforward,
            dropout=config.temporal_transformer_config.dropout,
            activation='relu',
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.temporal_transformer_config.num_layers
        )

        # 4. 位置编码
        seq_len = config.dataset_config.sequence_length
        self.positional_encoding = nn.Parameter(
            torch.randn(1, seq_len, config.point_mae_config.trans_dim) * 0.02
        )

        # 5. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(config.point_mae_config.trans_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, config.num_classes)
        )

    def load_model_from_ckpt(self, ckpt_path):
        """遵循项目原始风格的权重加载逻辑"""
        if ckpt_path is None:
            print_log('Training from scratch!!!', logger=self.logger_name)
            return

        print_log(f'Loading weights from {ckpt_path}...', logger=self.logger_name)
        state_dict = torch.load(ckpt_path, map_location='cpu')

        # 处理 DDP 包装
        if 'base_model' in state_dict:
            base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
        elif 'model' in state_dict:
            base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
        else:
            base_ckpt = state_dict

        # --- [原项目风格的核心逻辑] ---
        # 遍历所有 key，如果以前缀 MAE_encoder 开头，则重命名以匹配子模块
        for k in list(base_ckpt.keys()):
            if k.startswith('MAE_encoder.'):
                # 相当于去除 'MAE_encoder.' 前缀
                base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                del base_ckpt[k]
            # 过滤掉不需要的解码器部分，避免 unexpected keys 干扰
            elif k.startswith('MAE_decoder') or k.startswith('mask_token') or k.startswith('decoder_pos'):
                del base_ckpt[k]

        # 加载到子模块 point_mae_encoder 中
        incompatible = self.point_mae_encoder.load_state_dict(base_ckpt, strict=False)

        # 使用 utils.checkpoint 中的工具打印日志
        if incompatible.missing_keys:
            # 过滤掉我们知道会丢失的 cls_token 等，只关心 encoder 内部
            real_missing = [k for k in incompatible.missing_keys if 'encoder.' in k or 'blocks.' in k]
            if len(real_missing) > 0:
                print_log('missing_keys', logger=self.logger_name)
                print_log(get_missing_parameters_message(real_missing), logger=self.logger_name)

        # if incompatible.unexpected_keys:
        #     print_log('unexpected_keys', logger=self.logger_name)
        #     print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger=self.logger_name)

        print_log(f'[{self.logger_name}] Successful Loading the ckpt from {ckpt_path}', logger=self.logger_name)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder and mode:
            self.point_mae_encoder.eval()

    def get_loss_acc(self, ret, gt):
        logits = ret
        gt = gt.long()
        loss = torch.nn.functional.cross_entropy(logits, gt)
        _, pred = torch.max(logits, 1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def forward(self, point_sequence):
        # (B, T, N, 3)
        batch_size, seq_len, npoints, dims = point_sequence.shape
        point_sequence_flat = point_sequence.view(batch_size * seq_len, npoints, dims)

        # 提取特征
        frame_features = self.point_mae_encoder(point_sequence_flat, feature_only=True)
        feature_sequence = frame_features.view(batch_size, seq_len, -1)

        # 位置编码处理
        if seq_len <= self.positional_encoding.size(1):
            pos = self.positional_encoding[:, :seq_len, :]
        else:
            pos = self.positional_encoding[:, :self.positional_encoding.size(1), :]

        feature_sequence = feature_sequence + pos

        # Transformer
        transformer_out = self.temporal_transformer(feature_sequence)

        # 聚合 & 分类
        aggregated_feature = transformer_out.mean(dim=1)
        logits = self.classifier(aggregated_feature)
        return logits