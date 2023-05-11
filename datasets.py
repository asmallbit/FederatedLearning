

import torch 
import numpy as np
from torchvision import datasets, transforms
from dataset import dog_and_cat
from utils.utils import *

def get_dataset(dir, name):
	
	if name=='mnist':
		train_transform = transforms.Compose([
			transforms.RandomRotation(degrees=15),	# 随即旋转
			transforms.RandomCrop(28, padding=2),	# 随机裁剪
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))	# 归一化
		])

		test_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
		])

		train_dataset = datasets.MNIST(f'{dir}/mnist', train=True, download=True, transform=train_transform)
		eval_dataset = datasets.MNIST(f'{dir}/mnist', train=False, download=True, transform=test_transform)
		
	elif name=='cifar':
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),	# 随机裁剪
			transforms.RandomHorizontalFlip(),		# 随机翻转
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		
		train_dataset = datasets.CIFAR10(f'{dir}/cifar-10', train=True, download=True,
										transform=transform_train)
		eval_dataset = datasets.CIFAR10(f'{dir}/cifar-10', train=False, transform=transform_test)
		
	elif name == 'dog-and-cat':
		transform_train = transforms.Compose([
			transforms.RandomResizedCrop(224),		# 随机裁剪224*224
			transforms.RandomHorizontalFlip(),		# 随机水平翻转
			transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),	# 随机改变颜色
			transforms.RandomRotation(degrees=15),	# 随机旋转图像
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),		# 
		])

		transform_test = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		])

		train_dataset = dog_and_cat.DogCatDataset(f'{dir}/dog-and-cat', train=True, download=True,
										transform=transform_train)
		eval_dataset = dog_and_cat.DogCatDataset(f'{dir}/dog-and-cat', train=False, transform=transform_test)

	return train_dataset, eval_dataset

# 获取Non-IID数据
def split_non_iid(train_labels, alpha, n_clients):
	n_classes = train_labels.max() + 1
	label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
	class_idx = [np.argwhere(train_labels == y).flatten()
					for y in range(n_classes)]
	client_idx = [[] for _ in range(n_clients)]
	for c, f in zip(class_idx, label_distribution):
		for i, idx in enumerate(np.split(c, (np.cumsum(f)[:-1] * len(c)).astype(int))):
			client_idx[i] += [idx]
	client_idx = [np.concatenate(idx) for idx in client_idx]
	return client_idx


# 获取与训练集标签分布相同的测试集
def split_eval_dataset(train_labels, train_dataset, eval_dataset, split_idx):
	# 存储每个客户端的标签计数
	client_num = len(split_idx)
	n_classes = train_labels.max() + 1
	if torch.is_tensor(train_dataset.targets):
		train_label_counts = np.zeros((client_num, len(torch.unique(train_dataset.targets))))
		rate = np.zeros((client_num, len(torch.unique(train_dataset.targets))))	# 存储各个客户端的训练集某个标签数量占全部客户端这个标签数量的比例
	else:
		train_label_counts = np.zeros((client_num, len(np.unique(train_dataset.targets))))
		rate = np.zeros((client_num, len(np.unique(train_dataset.targets))))

	# 遍历每个客户端的数据集
	for i in range(client_num):
		for j in split_idx[i]:
			train_label_counts[i][train_dataset.targets[j]] += 1
	
	label_sum = [0 for _ in range(n_classes)]	# 所有客户端中各个标签数目之和
	
	eval_client_idx = [[] for _ in range(client_num)]
	label_indices = get_dataset_label_array(eval_dataset)
	samples = get_samples_by_label(eval_dataset)	# 存储测试集中, 每个标签对应的所有序号

	# 计算split_idx中各个标签的总数
	for i in range(n_classes):
		for j in range(client_num):
			label_sum[i] += train_label_counts[j][i]

	for i in range(client_num):
		for j in range(n_classes):
			rate[i][j] = train_label_counts[i][j] / label_sum[j]
			temp = np.random.choice(samples[j], 
									size=int(rate[i][j] * label_indices[j]), replace=False)
			eval_client_idx[i].extend(temp)
		eval_client_idx[i] = sorted(eval_client_idx[i])

	return eval_client_idx
			
# 统计数据集各个标签的样本数目
def get_dataset_label_array(dataset):
	if torch.is_tensor(dataset.targets):
		label_counts = [0 for _ in range(len(torch.unique(dataset.targets)))]
	else:
		label_counts = [0 for _ in range(len(np.unique(dataset.targets)))]

	for i in range(len(dataset)):
		label_counts[dataset.targets[i]] += 1
	return label_counts

# 获取数据集中标签所包含的所有元素对应的序号
def get_samples_by_label(dataset):
	if torch.is_tensor(dataset.targets):
		samples = [[] for _ in range(len(torch.unique(dataset.targets)))]
	else:
		samples = [[] for _ in range(len(np.unique(dataset.targets)))]
	for i in range(len(dataset)):
		samples[dataset.targets[i]].append(i)
	return samples

# 获取每个客户端的labels个数
def get_labels_num_each_client(datasets, array):
	if torch.is_tensor(datasets.targets):
		sample = [[0 for _ in range(len(torch.unique(datasets.targets)))] for _ in range(get_global_world_size())]
	else:
		sample = [[0 for _ in range(len(np.unique(datasets.targets)))] for _ in range(get_global_world_size())]
	for i in range(len(array)):
		for j in range(len(array[i])):
			sample[i][datasets.targets[array[i][j]]] += 1
	return sample