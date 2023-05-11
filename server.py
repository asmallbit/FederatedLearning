import models, torch
import copy
import datasets
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
from typing import Union
from torch.utils.data import DataLoader, Subset

from push.push import Push
from utils.distributed_utils import *

class Server(object):
	
	def __init__(self, conf, model, eval_dataset, train_dataset, push: Push,
					device: Union[int, str]="cpu"):
	
		self.conf = conf 
		
		self.global_model = copy.deepcopy(model)

		self.device = device

		self.push = push

		self.eval_dataset = eval_dataset

		self.train_dataset = train_dataset

		self.eval_split_idx = []

		# self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=False)

	def split_data(self):
		labels = np.array(self.train_dataset.targets)
		split_idx = datasets.split_non_iid(labels, alpha=self.conf["alpha"], n_clients=get_global_world_size())
		eval_split_idx = datasets.split_eval_dataset(labels, self.train_dataset, self.eval_dataset, split_idx)
		self.eval_split_idx = eval_split_idx	# 保存此项, 在模型评估阶段分配评估数据使用
		
		# 存取各个客户端中的标签对应的样本的数目
		train_dataset_labels_array = datasets.get_labels_num_each_client(self.train_dataset, split_idx)
		eval_dataset_labels_array = datasets.get_labels_num_each_client(self.eval_dataset, eval_split_idx)

		# 绘制训练集数据分布图像
		plt.clf()
		plt.figure(figsize=(20, 3))
		x = range(labels.max() + 1)
		colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
		plt.xticks(x)
		if get_global_world_size() == 1:
			plt.bar(x, train_dataset_labels_array[0], label="Client", color=colors[0])
		else:
			bottom = None
			for i in range(get_global_world_size()):
				plt.bar(x, train_dataset_labels_array[i], bottom=bottom, label=f"Client {i}", color=colors[i % len(colors)])
				if bottom is None:
					bottom = train_dataset_labels_array[i]
				else:
					bottom = np.add(train_dataset_labels_array[i], bottom)
		plt.legend()
		# 保存训练集数据分布图像
		train_data_path = f"./figures/{self.conf['type']}/{self.conf['model_name']}/train-data-distribution.png"
		os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
		plt.savefig(train_data_path)

		# 绘制测试集数据分布图像
		plt.clf()  # 清除上次绘制的内容
		plt.figure(figsize=(20, 3))
		plt.xticks(x)
		if get_global_world_size() == 1:
			plt.bar(x, eval_dataset_labels_array[0], label="Client", color=colors[0])
		else:
			bottom = None
			for i in range(get_global_world_size()):
				plt.bar(x, eval_dataset_labels_array[i], bottom=bottom, label=f"Client {i}", color=colors[i % len(colors)])
				if bottom is None:
					bottom = eval_dataset_labels_array[i]
				else:
					bottom = np.add(eval_dataset_labels_array[i], bottom)
		plt.legend()
		# 保存测试集数据分布图像
		eval_data_path = f"./figures/{self.conf['type']}/{self.conf['model_name']}/eval-data-distribution.png"
		os.makedirs(os.path.dirname(eval_data_path), exist_ok=True)
		plt.savefig(eval_data_path)

		return split_idx

	def calculate_weight_accumulator(self, models_params, samples):
		len_array = []
		total = 0
		for sample in samples:
			len_array.append(sample.shape[0])
			total += sample.shape[0]
		weight_accumulator = {}
		for model_params in models_params:
			# 聚合这些数据
			index = 0
			for key, value in model_params.items():
				if key in weight_accumulator:
					weight_accumulator[key] += value * (len_array[index] / total)
				else:
					weight_accumulator[key] = value * (len_array[index] / total)
		return weight_accumulator

	
	def model_aggregate(self, weight_accumulator):
		# 清除模型参数
		for key in self.global_model.state_dict():
			torch.nn.init.zeros_(self.global_model.state_dict()[key])
		
		# 更新模型参数
		for name, data in self.global_model.state_dict().items():
			
			update_per_layer = weight_accumulator[name] * self.conf["lambda"]
			data = data.float()
			update_per_layer = update_per_layer.float()
			data.add_(update_per_layer)
				
	def model_eval(self, group):
		self.global_model.eval()
		
		total_loss = 0.0
		correct = 0
		correct_k = 0
		dataset_size = 0

		# 合并group中包含的客户端分配的模型
		eval_dataset_split_idx = None
		if group is None or len(group) == 0:
			return 0.0, 0.0, 0.0	# 该簇为空
		elif len(group) == 1:
			eval_dataset_split_idx = self.eval_split_idx[group[0]]
		else:	# 两个及以上
				eval_dataset_split_idx = np.hstack(
    							tuple(self.eval_split_idx[group[i]] 
									for i in range(len(group))))

		# eval_loader每次都需要根据簇中的客户端重新生成
		eval_loader = DataLoader(dataset=Subset(self.eval_dataset, np.unique(np.sort(eval_dataset_split_idx))),
								batch_size=self.conf["batch_size"],
								shuffle=False,
								num_workers = mp.cpu_count(),
								prefetch_factor=2, pin_memory=True)
		
		for batch_id, batch in enumerate(eval_loader):
			data, target = batch
			data = data.to(self.device)
			target = target.to(self.device)

			dataset_size += data.size()[0]

			output = self.global_model(data)
			
			total_loss += torch.nn.functional.cross_entropy(output, target,
											  reduction='sum').item() # sum up batch loss
			pred = output.data.max(1)[1]  # get the index of the max log-probability
			correct += pred.eq(target.data.view_as(pred)).cpu().sum().float().item()
			# top5 correct
			target_resize = target.view(-1, 1)
			_, pred = output.topk(5)
			correct_k += torch.eq(pred, target_resize).cpu().sum().float().item()

		acc = correct / dataset_size
		acc5 = correct_k / dataset_size
		loss = total_loss / dataset_size

		return acc, acc5, loss