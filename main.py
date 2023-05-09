import argparse, json
import datetime
import os
import logging
import numpy as np
import pickle
import sys
import time
import torch, random
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

from server import *
from client import *
import models, datasets
from push.push import Push
from utils.distributed_utils import *
from utils.output_handler import *
from utils.utils import *

def main(conf, args):
	seed_everything(5)
	init_distributed_mode() # 进程组初始化
	local_rank = get_local_rank()
	global_rank = get_global_rank()
	local_world_size = get_local_world_size()
	global_world_size = get_global_world_size()
	k = conf["k"]

	gpu = args.gpu

	push = Push(conf)		# Push消息

	if local_world_size != 1:
		device = torch.device("cuda:{}".format(local_rank))
	elif gpu == 0:
		device = torch.device("cuda:0") # 只有1个GPU
	else:
		device = torch.device("cpu")

	if str(device) == "cpu":
		notify_user("[Client " + str(global_rank) + "] GPU is not enabled", push)
	else:
		notify_user("[Client " + str(global_rank) + "] GPU is using GPU " + str(local_rank) + " now", push)

	train_datasets, eval_datasets = datasets.get_dataset("./data", conf["type"])
	global_model = models.get_model(conf["model_name"], True, device) # Set the flag to True to get pretrained model
	servers = [Server(conf, global_model, eval_datasets, push, device) for _ in range(k)]	# 创建k个server, 相互独立但初始值相同

	dataset_split_idx = None
	dataset_split_idx_size = 0
	split_idx = servers[0].split_data()

	if is_global_main_process():
		dataset_split_idx = split_idx[0]
		if global_world_size > 1:
			for num in range(1, global_world_size):
				tensor = torch.tensor([split_idx[num].shape[0]])
				dist.send(tensor, dst=num)
	else:
		tensor = torch.tensor([0])
		dist.recv(tensor, src=0)
		dataset_split_idx_size = tensor[0].item()
	dist.barrier()

	if is_global_main_process():
		if global_world_size > 1:
			for num in range(1, global_world_size):
				tensor = torch.from_numpy(split_idx[num])
				dist.send(tensor, dst=num)
	else:
		tensor = torch.tensor([0 for i in range(dataset_split_idx_size)])
		dist.recv(tensor, src=0)
		dataset_split_idx = tensor.numpy()
	dist.barrier()

	diffs = [dict() for _ in range(global_world_size + k)]

	if is_global_main_process():
		# 分别用来存储全局epoch的acc和loss
		acc_list = [[] for _ in range(k)]
		loss_list = [[] for _ in range(k)]
		# 分别用来存储每个客户端本地epoch的acc和loss
		client_acc_list = [[] for _ in range(global_world_size)]
		client_loss_list = [[] for _ in range(global_world_size)]
		# 服务端推送消息
		message = (f"************************  TASK STARTED  ************************\nModel: {conf['model_name']}\n"
				f"Dataset: {conf['type']}\nBatch Size: {conf['batch_size']}\nNumber of Clients: {global_world_size}\n"
				f"Learning rate: {conf['lr']}\nMomentum: {conf['momentum']}\nK: {k}\nFactor: {conf['factor']}\n"
				f"Patience: {conf['patience']}\nGlobal Epochs: {conf['global_epochs']}\n"
				f"Local Epochs: {conf['local_epochs']}")

		notify_user(message, push)
		print("\n")
	client = Client(conf, global_model, train_datasets, eval_datasets, push, dataset_split_idx, device)
	dist.barrier() # 确保客户端全部创建完毕

	for e in range(conf["global_epochs"]):		# global epochs 全局轮次
		if is_global_main_process():
			# 每一轮epoch开始时提示
			message = "==============  Global Epoch " + str(e) + "  =============="
			notify_user(message, push)
			message = "[Global Model] Ephch " + str(e) + " started"
			notify_user(message, push)
		else:
			# 在local_rank的非根节点的terminal打印消息, 否则客户端的terminal并不能看到是第几轮训练
			message = "==============  Global Epoch " + str(e) + "  =============="
			print(message)
			message = "[Global Model] Ephch " + str(e) + " started"
			print(message)

		model_params = client.local_train()
		dist.barrier()	# 训练完毕
		# 客户端向服务端发送本地训练后的模型和全局模型的diff, 服务端接收客户端发送的数据并保存
		if is_global_main_process():
			diffs[0] = model_params
			# 收集各个客户端的diff
			for key, _ in global_model.state_dict().items():
				for i in range(1, global_world_size):
					temp = torch.zeros_like(model_params[key])
					dist.recv(temp, src=i)
					diffs[i][key] = temp
		else:
			# 向服务端发送diff各个key对应的value
			for key, value in model_params.items():
				dist.send(value, dst=0)
		dist.barrier()

		# 用区分rank为0的进程, send和recv不能用于同一个进程
		# ValueError: Invalid destination rank: destination rank should not be the same as the rank of the current process.
		if is_global_main_process():
			# 收集每个客户端发送来的数据
			if global_world_size > 1:
				for i in range(1, global_world_size):
					tensor = torch.zeros(1, 2)
					dist.recv(tensor, src=i)
					client_acc_list[i].append(tensor[0][0].item())
					client_loss_list[i].append(tensor[0][1].item())

			main_client_acc, main_client_loss = client.model_eval(e)
			client_acc_list[0].append(main_client_acc)
			client_loss_list[0].append(main_client_loss)
		else:
			client.model_eval(e)

		if is_global_main_process():
			# 模型参数聚合
			# 根据diffs数组中的值对diff进行聚合
			'''
			这里我们将k个全局模型也添加到diffs里，一起作为kmeans的输入，更新全局模型时，我们只需要将该全局模型所在簇
			的模型参数使用FedAvg方法进行合并，添加到该全局模型的参数中完成更新
			'''
			# 将k个全局模型也添加到diffs
			for i in range(k):
				diffs[global_world_size + i] = servers[i].global_model.state_dict()
			# 聚合
			kmeans = cluster_kmeans(diffs, k)
			kmeans_grouped = {}	# 存储各个客户端与全局模型的对应关系
			weight_accumulators = [[] for _ in range(k)] # 用于存储每组模型参数的数组
			global_model_dict = {}
			for i, label in enumerate(kmeans.labels_):
				if i < global_world_size:
					weight_accumulators[label].append(diffs[i]) # 只添加非全局模型
				else:
					global_model_dict[i - global_world_size] = label
				if label in kmeans_grouped:
					kmeans_grouped[label].append(i)
				else:
					kmeans_grouped[label] = [i]

			# 用于存储全局模型对应的本地模型，主要用来在terminal显示客户端所属簇和判断全局模型所在簇的客户端模型
			global_models_grouped = [[] for _ in range(k)]
			# 遍历kmeans_grouped, 找到每个全局模型所在的簇
			for labels, group in kmeans_grouped.items():
				for item in group:
					if item >= global_world_size:
						# 找到对应的元素
						global_models_grouped[item - global_world_size] = group

			# kmeans_grouped进行修改，去掉值大于等于global_world_size的元素, 也就是删除全局模型对应索引
			for labels, group in kmeans_grouped.items():
				temp = 0
				while temp < len(group):
					if group[temp] >= global_world_size:
						del group[temp]
					else:
						temp += 1

			for i in range(k):
				if len(global_models_grouped[i]) == 0:
					notify_user(f"Global Group {i} contains nothing", push)
				else:
					notify_user(f"Global Group {i} contains client {global_models_grouped[i]}", push)
			notify_user("[Global Model] Aggregating the global model from trained local models", push)

			# 更新k个全局模型
			for i in range(k):
				if len(global_models_grouped) != 0:	# 仅在此全局模型所在簇有其他客户端模型时才进行更新
					servers[i].model_aggregate(
						servers[i].calculate_weight_accumulator(weight_accumulators[global_model_dict[i]], split_idx))	# 更新n个全局模型
				acc, acc_5, loss = servers[i].model_eval()
				# Append accuracy and loss for this epoch to the corresponding lists
				acc_list[i].append(acc)
				loss_list[i].append(loss)
				message = f"Global Model {i}: acc = {acc} , acc_5 = {acc_5} loss = {loss}"
				notify_user(message, push)
			notify_user("[Global Epoch] Epoch " + str(e) + " done!", push)

			# 将更新后的全局模型的参数分发到各个节点
			for key, values in kmeans_grouped.items():	# 这里的key是k-means结果中客户端所属的组, value是各个客户端节点的global rank值
				for value in values:
					if value != 0:
						# 将diffs的第{key}个值发送到客户端{value}
						for param, data in diffs[key].items():
							dist.send(data, dst=value)
					else:
						# 更新rank为0的客户端的数据
						client.update_model(diffs[key])		
		else:
			# 接收服务器发来的参数
			for key, value in model_params.copy().items():	# 更新diff的值
				temp = torch.zeros_like(model_params[key])
				dist.recv(temp, src=0)
				model_params[key] = temp

			# 更新本地模型的值
			client.update_model(model_params)

		dist.barrier()

	# 只需要服务器去处理保存模型和保存accuracy/loss随epoch次数的变化图的操作
	if is_global_main_process():
		# 保存模型
		type = conf["type"]
		model_name = conf["model_name"]
		path = "./result/" + type + "/" + model_name + "/"
		if not os.path.isdir(path):
			os.makedirs(path)
		for i in range(k):
			torch.save(servers[i].global_model.state_dict(), path + type + "-" + model_name + f"-{i}.pth")
		
		# 绘制准确率图像
		colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
		path = "./figures/" + type + "/" + model_name + "/"
		if not os.path.isdir(path):
			os.makedirs(path)
		for acc in acc_list:
			plt.plot(range(conf["global_epochs"]), acc, label=f'Global Model{i} Accuracy', linewidth=1.0, color=colors[i % len(colors)])
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
		plt.legend()
		plt.savefig(path + type + "-" + model_name + "-global-accuracy.png")
		plt.show()

		# 绘制误差图像
		plt.clf()
		for loss in loss_list:
			plt.plot(range(conf["global_epochs"]), loss, label=f'Global Model{i} Loss', linewidth=1.0, color=colors[i % len(colors)])
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig(path + type + "-" + model_name + "-global-loss.png")
		plt.show()

		# 绘制各个客户端的accuracy图像
		plt.clf()
		i = 0
		for acc in client_acc_list:
			plt.plot(range(conf["global_epochs"]), acc, label=f"Client{i} Accuracy", linewidth=1.0, color=colors[i % len(colors)])
			i += 1
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
		plt.legend()
		plt.savefig(path + type + "-" + model_name + "-client-accuracy.png")
		plt.show()

		# 绘制各个客户端的loss图像
		plt.clf()
		i = 0
		for loss in client_loss_list:
			plt.plot(range(conf["global_epochs"]), loss, label=f"Client{i} Loss", linewidth=1.0, color=colors[i % len(colors)])
			i += 1
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig(path + type + "-" + model_name + "-client-loss.png")
		plt.show()

	# Shutdown
	dist.destroy_process_group()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', type=str, default='./utils/conf.json', dest='conf')
	parser.add_argument('--gpu', type=int, default=-1, help='Which GPU to run,-1 mean CPU, 0,1,2,... for GPU')
	args  = parser.parse_args()

	with open(args.conf, 'r') as f:
		conf = json.load(f)

	main(conf, args)
