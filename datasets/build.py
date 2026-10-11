from utils import registry

# "工具房"只需要创建好"登记簿"(DATASETS)
DATASETS = registry.Registry('dataset')


# 和"工厂函数"(build_dataset_from_cfg)
def build_dataset_from_cfg(cfg, default_args=None):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        cfg (eDICT):
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    # DEBUG: 添加调试信息
    if default_args is not None:
        print(f"[DEBUG build_dataset_from_cfg] cfg.NAME = {getattr(cfg, 'NAME', 'NOT_FOUND')}")
        print(f"[DEBUG build_dataset_from_cfg] default_args.subset = {getattr(default_args, 'subset', 'NOT_FOUND')}")
    else:
        print(f"[DEBUG build_dataset_from_cfg] default_args is None!")

    # 确保配置正确合并
    if default_args is not None:
        # 将 default_args 的属性合并到 cfg 中（如果需要）
        for key, value in default_args.__dict__.items():
            if not hasattr(cfg, key):
                setattr(cfg, key, value)
    result = DATASETS.build(cfg, default_args=default_args)
    print(f"[DEBUG build_dataset_from_cfg] Built dataset: {type(result)}, length: {len(result)}")
    return result
