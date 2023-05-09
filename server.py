import models, torch
import datasets
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Union

from push.push import Push
from utils.distributed_utils import *

class Server(object):
	
	def __init__(self, conf, model, eval_dataset, push: Push, 
					device: Union[int, str]="cpu"):
	
		self.conf = conf 
		
		self.global_model = model

		self.device = device

		self.push = push
		
		self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=False)

	def split_data(self):
		train_datasets, eval_datasets = datasets.get_dataset("./data", self.conf["type"])
		labels = np.array(train_datasets.targets)
		split_idx = datasets.split_non_iid(labels, alpha=self.conf["alpha"], n_clients=get_global_world_size())
		plt.figure(figsize=(20, 3))
		plt.hist([labels[idc] for idc in split_idx], stacked=True,
					bins=np.arange(min(labels) - 0.5, max(labels) + 1.5, 1),
					label=[f"Client{i}" for i in range(get_global_world_size())], rwidth=0.5)
		plt.xticks(np.arange(len(train_datasets.classes)), np.arange(0, len(train_datasets.classes)))
		plt.legend()
		path = f"./figures/{self.conf['type']}/{self.conf['model_name']}"
		if not os.path.isdir(path):
			os.makedirs(path)
		plt.savefig(f"{path}/data-distribution.png")
		return split_idx

	def calculate_weight_accumulator(self, models_params):
		weight_accumulator = {}
		for model_params in models_params:
			# 聚合这些数据
			for key, value in model_params.items():
				if key in weight_accumulator:
					weight_accumulator[key] += (value - self.global_model.state_dict()[key]) / len(models_params)
				else:
					weight_accumulator[key] = (value - self.global_model.state_dict()[key]) / len(models_params)
		return weight_accumulator

	
	def model_aggregate(self, weight_accumulator):
		for name, data in self.global_model.state_dict().items():
			
			update_per_layer = weight_accumulator[name] * self.conf["lambda"]
			data = data.float()
			update_per_layer = update_per_layer.float()
			data.add_(update_per_layer)
				
	def model_eval(self):
		self.global_model.eval()
		
		total_loss = 0.0
		correct = 0
		dataset_size = 0
		for batch_id, batch in enumerate(self.eval_loader):
			data, target = batch
			data = data.to(self.device)
			target = target.to(self.device)

			dataset_size += data.size()[0]

			output = self.global_model(data)
			
			total_loss += torch.nn.functional.cross_entropy(output, target,
											  reduction='sum').item() # sum up batch loss
			pred = output.data.max(1)[1]  # get the index of the max log-probability
			correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

		acc = float(correct) / float(dataset_size)
		loss = total_loss / dataset_size

		return acc, loss