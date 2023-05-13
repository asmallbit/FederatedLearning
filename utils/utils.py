import torch
import itertools
import numpy as np
import random
import matplotlib.pyplot as plt
from utils.distributed_utils import *
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Current GPU
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed) # All GPU (Optional)

def cluster_kmeans(dicts, k, epoch, model_name, type):
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
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)    # 将data降至2维

    # 使用kmeans对矩阵进行聚类
    kmeans = KMeans(n_clusters=k, init="k-means++", algorithm="elkan", random_state=5)
    label = kmeans.fit_predict(reduced_data)

    # 画图
    centroids = kmeans.cluster_centers_

    plt.clf()
    for i in np.unique(label):
        plt.scatter(reduced_data[label == i, 0], reduced_data[label == i, 1], label = f"Group {i}")
    # plt.scatter(centroids[:,0], centroids[:,1], c="black")   # 打印各聚簇中心点
    plt.legend()
    path = f"./figures/{type}/{model_name}/cluster/"
    if not os.path.isdir(path):
	    os.makedirs(path)
    plt.savefig(f"{path}{type}-{model_name}-cluster-{epoch + 1}.png")

    return kmeans.fit(reduced_data)

# 根据给定的数量生成若干种颜色
def get_colors_array(client_num):
    # 定义自定义的颜色循环
    colors = plt.cm.tab20(np.linspace(0, 1, client_num))  # 生成20个不重复的颜色
    colors_cycle = itertools.cycle(colors)
    return colors_cycle 
