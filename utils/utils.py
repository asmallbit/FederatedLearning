import torch
import copy
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

def cluster_kmeans(dicts, k, original, epoch, model_name, type):
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
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=5)
    label = copy.deepcopy(kmeans.fit_predict(reduced_data))

    # 查看本次和上次的聚类结果是否相同, 如果不同很大概率是有新设备加入了
    # mapping 存储的是新的标签和旧的标签的对应关系, k-means每次返回的结果并非每次顺序都是一样的
    # eg 上一次可能是{0:{1, 2}, 1:{0}} 本次为 {0:{0}, 1:{1, 2}}, 我们仍然将它们视为相同, 但是它们的键是存在一个映射关系{0:1, 1:0}的
    is_same, mapping = is_same_result(original, kmeans_result_2_dict(kmeans.fit(reduced_data)))

    if mapping is not None: # mapping不为None, 也就是is_same为True
        # 按照mapping新旧字典的对应关系修改label
        for i in range(len(label)):
            label[i] = mapping[label[i]]

    # 画图
    # centroids = kmeans.cluster_centers_

    # 聚类结果图
    plt.clf()
    for i in np.unique(label):
        plt.scatter(reduced_data[label == i, 0], reduced_data[label == i, 1], label = f"Group {i}")
    # plt.scatter(centroids[:,0], centroids[:,1], c="black")   # 打印各聚簇中心点
    plt.legend()
    path = f"./figures/{type}/{model_name}/cluster/"
    if not os.path.isdir(path):
	    os.makedirs(path)
    plt.savefig(f"{path}{type}-{model_name}-cluster-{epoch + 1}.png")

    # 误差平方图
    plt.clf()
    sse = []
    for i in range(1, get_global_world_size()):
        temp = KMeans(n_clusters=i)
        temp.fit(reduced_data)
        sse.append(temp.inertia_)
    plt.xlabel("k")
    plt.ylabel("SSE")
    plt.plot(range(1, get_global_world_size()), sse, 'o-')    # 点线图
    path = f"./figures/{type}/{model_name}/cluster/sse/"
    if not os.path.isdir(path):
	    os.makedirs(path)
    plt.savefig(f"{path}{type}-{model_name}-cluster-sse-{epoch + 1}.png")

    if is_same:
        return True, original
    elif epoch == 0:            # 第一轮全局训练
        return True, kmeans_result_2_dict(kmeans.fit(reduced_data))
    else:
        return False, kmeans_result_2_dict(kmeans.fit(reduced_data))      # 如果不同, 我们在本轮结束时, 对k个全局模型进行一次聚合, 增加模型的泛化性

# 根据给定的数量生成若干种颜色
def get_colors_array(client_num):
    # 定义自定义的颜色循环
    colors = plt.cm.tab20(np.linspace(0, 1, client_num))  # 生成client_num个不重复的颜色
    colors_cycle = itertools.cycle(colors)
    return colors_cycle

def is_same_result(old_result, new_result):
    '''比较old_result和new_result两个字典中的值List是否相同, 不考虑List的顺序以及List内元素的顺序
    Arg:
        old_result: 上一轮聚类的结果
        new_result: 本轮聚类的结果
    '''
    if old_result is None or new_result is None:
        return False, None

    if set(old_result.keys()) != set(new_result.keys()):
        return False, None
    
    mapping = {}

    for key_old in old_result.keys():
        temp = 0
        for key_new in new_result.keys():
            if sorted(old_result[key_old]) == sorted(new_result[key_new]):
                mapping[key_new] = key_old
                break
            temp += 1
            if temp == len(new_result.keys()):
                return False, None
    return True, mapping

# 将k-means的结果转化为字典
def kmeans_result_2_dict(result):
    kmeans_grouped = {}	# 存储各个客户端与全局模型的对应关系
    for i, label in enumerate(result.labels_):
        if label in kmeans_grouped:
            kmeans_grouped[label].append(i)
        else:
            kmeans_grouped[label] = [i]
    return kmeans_grouped