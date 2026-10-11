import torch
import torch.nn as nn
from .build import MODELS
from tools import builder
from utils.logger import print_log
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message


@MODELS.register_module()
class PointMAE_GRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logger_name = 'PointMAE_GRU'
        print_log(f'[{self.logger_name}] Building model...', logger=self.logger_name)

        self.config = config

        self.point_mae_encoder = builder.model_builder(config.point_mae_config)

        self.freeze_encoder = config.get('freeze_encoder', True)
        if self.freeze_encoder:
            print_log(f'[{self.logger_name}] Freezing Point-MAE encoder parameters.', logger=self.logger_name)
            for param in self.point_mae_encoder.parameters():
                param.requires_grad = False

        self.gru = nn.GRU(
            input_size=config.point_mae_config.trans_dim,
            hidden_size=config.gru_hidden_size,
            num_layers=config.gru_num_layers,
            batch_first=True,
            bidirectional=config.get('bidirectional', False)
        )

        gru_output_dim = config.gru_hidden_size * 2 if config.get('bidirectional', False) else config.gru_hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(gru_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, config.num_classes)
        )

    def load_model_from_ckpt(self, ckpt_path):
        if ckpt_path is None:
            return
        print_log(f'Loading weights from {ckpt_path}...', logger=self.logger_name)
        state_dict = torch.load(ckpt_path, map_location='cpu')

        if 'base_model' in state_dict:
            base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
        elif 'model' in state_dict:
            base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
        else:
            base_ckpt = state_dict

        for k in list(base_ckpt.keys()):
            if k.startswith('MAE_encoder.'):
                base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                del base_ckpt[k]
            elif k.startswith('MAE_decoder') or k.startswith('mask_token') or k.startswith('decoder_pos'):
                del base_ckpt[k]

        incompatible = self.point_mae_encoder.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            real_missing = [k for k in incompatible.missing_keys if 'encoder.' in k or 'blocks.' in k]
            if len(real_missing) > 0:
                print_log('missing_keys', logger=self.logger_name)
                print_log(get_missing_parameters_message(real_missing), logger=self.logger_name)

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
        batch_size, seq_len, npoints, dims = point_sequence.shape
        point_sequence_flat = point_sequence.view(batch_size * seq_len, npoints, dims)

        frame_features = self.point_mae_encoder(point_sequence_flat, feature_only=True)
        feature_sequence = frame_features.view(batch_size, seq_len, -1)

        gru_out, _ = self.gru(feature_sequence)
        last_time_step_out = gru_out[:, -1, :]
        logits = self.classifier(last_time_step_out)
        return logits