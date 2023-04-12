import os
import torch.distributed as dist

def init_distributed_mode():
    dist.init_process_group(backend="gloo",
                            timeout=datetime.timedelta(seconds=3600))   # 设置超时时间

# From https://github.com/pytorch/vision/blob/main/references/classification/utils.py
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_global_main_process(rank):
    return rank == 0