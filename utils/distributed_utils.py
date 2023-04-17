import datetime
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

def get_local_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return int(os.environ["LOCAL_WORLD_SIZE"])

def get_global_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return int(os.environ["WORLD_SIZE"])

def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ["LOCAL_RANK"])

def get_global_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ["RANK"])

def is_local_main_process():
    return get_local_rank() == 0

def is_global_main_process():
    return get_global_rank() == 0