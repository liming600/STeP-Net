import numpy as np
import torch
import random


class PointcloudRotate(object):
    def __call__(self, pc):
        # pc: (B, N, 3) or (B, T, N, 3)
        ndim = pc.dim()
        bsize = pc.size(0)

        for i in range(bsize):
            # --- [一致性关键] ---
            # 在循环内部生成一次随机参数，确保该样本（无论是否有时序）使用同一组参数
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            # 转换为 Tensor 并移动到对应设备
            R = torch.from_numpy(rotation_matrix.astype(np.float32)).to(pc.device)

            if ndim == 3:  # 单帧: (B, N, 3)
                pc[i, :, :] = torch.matmul(pc[i], R)
            elif ndim == 4:  # 时序: (B, T, N, 3)
                # 对该序列的所有帧应用相同的旋转矩阵 R
                for t in range(pc.size(1)):
                    pc[i, t, :, :] = torch.matmul(pc[i, t], R)

        return pc


class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        ndim = pc.dim()
        bsize = pc.size(0)

        for i in range(bsize):
            # --- [一致性关键] 生成一次参数 ---
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])

            scale = torch.from_numpy(xyz1).float().to(pc.device)
            translate = torch.from_numpy(xyz2).float().to(pc.device)

            if ndim == 3:
                pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], scale) + translate
            elif ndim == 4:
                for t in range(pc.size(1)):
                    # 所有帧应用相同的缩放和平移
                    pc[i, t, :, 0:3] = torch.mul(pc[i, t, :, 0:3], scale) + translate

        return pc


class PointcloudJitter(object):
    """
    注意：Jitter 通常是模拟传感器噪声。
    对于时序数据，通常每帧的噪声应该是独立的（Independent），而不是一致的（Consistent）。
    如果每帧噪声一模一样，看起来就像传感器上有个固定的污点，这不符合物理规律。
    因此，这里我们保持每帧独立生成噪声。
    """

    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, pc):
        # 无论是 3D 还是 4D，我们都希望对所有点施加随机噪声
        # 可以利用 Tensor 的广播机制或 flatten 处理，这里为了清晰使用循环或 view

        if pc.dim() == 3:
            bsize = pc.size(0)
            for i in range(bsize):
                jittered_data = pc.new(pc.size(1), 3).normal_(
                    mean=0.0, std=self.std
                ).clamp_(-self.clip, self.clip)
                pc[i, :, 0:3] += jittered_data

        elif pc.dim() == 4:
            # (B, T, N, 3) -> 视为 (B*T, N, 3) 统一加噪声，实现每帧独立
            B, T, N, C = pc.shape
            pc_flat = pc.view(-1, N, C)
            jittered_data = pc.new(B * T, N, 3).normal_(
                mean=0.0, std=self.std
            ).clamp_(-self.clip, self.clip)
            pc_flat[:, :, 0:3] += jittered_data
            # view 操作是共享内存的，所以修改 pc_flat 就是修改 pc

        return pc


class PointcloudScale(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        ndim = pc.dim()
        bsize = pc.size(0)

        for i in range(bsize):
            # --- [一致性关键] ---
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            scale = torch.from_numpy(xyz1).float().to(pc.device)

            if ndim == 3:
                pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], scale)
            elif ndim == 4:
                for t in range(pc.size(1)):
                    pc[i, t, :, 0:3] = torch.mul(pc[i, t, :, 0:3], scale)

        return pc


class PointcloudTranslate(object):
    def __init__(self, translate_range=0.2):
        self.translate_range = translate_range

    def __call__(self, pc):
        ndim = pc.dim()
        bsize = pc.size(0)

        for i in range(bsize):
            # --- [一致性关键] ---
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            translate = torch.from_numpy(xyz2).float().to(pc.device)

            if ndim == 3:
                pc[i, :, 0:3] = pc[i, :, 0:3] + translate
            elif ndim == 4:
                for t in range(pc.size(1)):
                    pc[i, t, :, 0:3] = pc[i, t, :, 0:3] + translate

        return pc


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.5):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, pc):
        # Dropout 类似于 Jitter，通常每帧独立随机丢弃，模拟遮挡或稀疏变化
        ndim = pc.dim()
        bsize = pc.size(0)

        if ndim == 3:
            for i in range(bsize):
                self._dropout_single_frame(pc[i])
        elif ndim == 4:
            for i in range(bsize):
                for t in range(pc.size(1)):
                    self._dropout_single_frame(pc[i, t])

        return pc

    def _dropout_single_frame(self, cur_pc):
        # 辅助函数：处理单帧 (N, 3)
        dropout_ratio = np.random.random() * self.max_dropout_ratio
        drop_idx = np.where(np.random.random((cur_pc.size(0))) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            # 将丢弃的点设置为第一个点的坐标（一种常见的填充方式，或者置零）
            cur_pc[drop_idx.tolist(), 0:3] = cur_pc[0, 0:3].repeat(len(drop_idx), 1)


class RandomHorizontalFlip(object):
    def __init__(self, upright_axis='z', is_temporal=False):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.is_temporal = is_temporal
        self.D = 4 if is_temporal else 3
        self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
        self.horz_axes = set(range(self.D)) - set([self.upright_axis])

    def __call__(self, coords):
        ndim = coords.dim()
        bsize = coords.size(0)

        for i in range(bsize):
            # --- [一致性关键] ---
            if random.random() < 0.95:
                for curr_ax in self.horz_axes:
                    if random.random() < 0.5:
                        # 确定要翻转，接下来计算 max 值并翻转

                        if ndim == 3:
                            coord_max = torch.max(coords[i, :, curr_ax])
                            coords[i, :, curr_ax] = coord_max - coords[i, :, curr_ax]
                        elif ndim == 4:
                            # 时序数据：必须基于整个序列的最大值，或者第一帧的最大值进行翻转？
                            # 通常为了保持空间一致性，我们计算整个序列在该轴的最大值，或者简单地使用统一逻辑

                            # 方法：计算整个序列 (T*N) 在该轴的最大值，确保翻转轴心一致
                            coord_max = torch.max(coords[i, :, :, curr_ax])

                            for t in range(coords.size(1)):
                                coords[i, t, :, curr_ax] = coord_max - coords[i, t, :, curr_ax]

        return coords