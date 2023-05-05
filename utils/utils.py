import torch
import numpy as np
import random
from utils.distributed_utils import *
from sklearn.cluster import KMeans

def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Current GPU
        torch.backends.cudnn.benchmark = False    # Close optimization
        torch.backends.cudnn.deterministic = True # Close optimization
        torch.cuda.manual_seed_all(seed) # All GPU (Optional)

def cluster_kmeans(dicts, k):
    # 将每个字典中的tensor值转换为numpy数组
    array_list = [[] for i in range(get_global_world_size())]    

    i = 0
    for dict in dicts:
        for name in dict:
            temp = dict[name].view(-1).tolist()
            array_list[i].extend(temp)
        i += 1

    # 将数组放入一个矩阵中
    data = np.vstack(array_list)

    # 使用kmeans对矩阵进行聚类
    return KMeans(n_clusters=k, random_state=0).fit(data)