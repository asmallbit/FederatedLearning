import argparse
import copy
import json
import os
import numpy as np
import torch
import torch.distributed as dist
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

	gpu = args.gpu
	k = conf["k"]

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
	servers = [Server(conf, global_model, eval_datasets, train_datasets, push, device) for _ in range(k)]	# 创建k个server, 相互独立但初始值相同

	dataset_split_idx = None
	dataset_split_idx_size = 0
	split_idx = servers[0].split_data()
	if global_world_size > 1:
		for i in range(1, k):
			servers[i].eval_split_idx = servers[0].eval_split_idx

	if is_global_main_process():
		dataset_split_idx = split_idx[0]
		if global_world_size > 1:
			for num in range(1, global_world_size):
				tensor = torch.tensor([split_idx[num].shape[0]], device=device)
				dist.send(tensor, dst=num)
	else:
		tensor = torch.tensor([0], device=device)
		dist.recv(tensor, src=0)
		dataset_split_idx_size = tensor[0].item()
	dist.barrier()

	if is_global_main_process():
		if global_world_size > 1:
			for num in range(1, global_world_size):
				tensor = torch.from_numpy(split_idx[num])
				dist.send(tensor, dst=num)
	else:
		tensor = torch.tensor([0 for i in range(dataset_split_idx_size)], device=device)
		dist.recv(tensor, src=0)
		dataset_split_idx = tensor.numpy()
	dist.barrier()

	diffs = [dict() for _ in range(global_world_size)]

	if is_global_main_process():
		# 分别用来存储全局epoch的acc和loss
		acc_list = [[] for _ in range(k)]
		loss_list = [[] for _ in range(k)]
		# 分别用来存储每个客户端本地epoch的acc和loss
		# client_acc_list = [[] for _ in range(global_world_size)]
		client_loss_list = [[] for _ in range(global_world_size)]
		# 服务端推送消息
		message = (f"************************  TASK STARTED  ************************\nModel: {conf['model_name']}\n"
				f"Dataset: {conf['type']}\nBatch Size: {conf['batch_size']}\nNumber of Clients: {global_world_size}\n"
				f"Learning rate: {conf['lr']}\nK: {k}\nGlobal Epochs: {conf['global_epochs']}\n"
				f"Local Epochs: {conf['local_epochs']}")

		notify_user(message, push)
		print("\n")
	client = Client(conf, global_model, train_datasets, eval_datasets, push, dataset_split_idx, device)
	dist.barrier() # 确保客户端全部创建完毕

	for e in range(conf["global_epochs"]):		# global epochs 全局轮次
		if is_global_main_process():
			# 每一轮epoch开始时提示
			message = "==============  Global Epoch " + str(e + 1) + "  =============="
			notify_user(message, push)
			message = "[Global Model] Ephch " + str(e + 1) + " started"
			notify_user(message, push)
		else:
			# 在local_rank的非根节点的terminal打印消息, 否则客户端的terminal并不能看到是第几轮训练
			message = "==============  Global Epoch " + str(e + 1) + "  =============="
			print(message)
			message = "[Global Model] Ephch " + str(e + 1) + " started"
			print(message)
		dist.barrier()	# 防止local_train打印到全局训练轮数提示的上面

		model_params, client_loss = client.local_train()

		# 接收loss
		# 用区分rank为0的进程, send和recv不能用于同一个进程
		# ValueError: Invalid destination rank: destination rank should not be the same as the rank of the current process.
		if is_global_main_process():
			# 收集每个客户端发送来的数据
			if global_world_size > 1:
				for clinet_index in range(1, global_world_size):
						tensor = torch.tensor([0.0, 0.0], device=device)
						dist.recv(tensor, src=clinet_index)
						# client_acc_list/client_loss_client的长度为e
						# client_acc_list[clinet_index].append(tensor[0].item())	# 暂时不统计本地训练accuracy
						client_loss_list[clinet_index].append(tensor[1].item())

			# client_acc_list[0].append(0)
			client_loss_list[0].append(client_loss)
		else:
			tensor = torch.tensor([0, client_loss], device=device)
			dist.send(tensor, dst=0)
		dist.barrier()

		# 客户端向服务端发送本地训练后的模型和全局模型的diff, 服务端接收客户端发送的数据并保存
		if is_global_main_process():
			diffs[0] = model_params

			# 收集各个客户端的diff
			if global_world_size > 1:
				for key, _ in servers[0].global_model.state_dict().items():
					for i in range(1, global_world_size):
						temp = torch.zeros_like(model_params[key])
						dist.recv(temp, src=i)
						diffs[i][key] = temp
		else:
			# 向服务端发送diff各个key对应的value
			for key, value in model_params.items():
				dist.send(value, dst=0)
		dist.barrier()

		if is_global_main_process():
			# 模型参数聚合
			# 根据diffs数组中的值对diff进行聚合
			kmeans = cluster_kmeans(diffs, k, e, conf["model_name"], conf["type"], push)	# 因为要画图，所以收集的参数多一些，核心参数只需要diffs和k
			kmeans_grouped = {}	# 存储各个客户端与全局模型的对应关系
			for i, label in enumerate(kmeans.labels_):
				if label in kmeans_grouped:
					kmeans_grouped[label].append(i)
				else:
					kmeans_grouped[label] = [i]
			
			# 发送通知, 通知kmeans的分类结果
			for i in range(k):
				if len(kmeans_grouped[i]) == 0:
					notify_user(f"WARNING: Global Group {i} contains NOTHING!!!!!", push)
				else:
					notify_user(f"Global Group {i} contains client {kmeans_grouped[i]}", push)
			
			client_dataset_items = [[] for _ in range(k)]	# 各个集群中各个客户端数据集中包含的数据标号
			# 将diffs按照聚类结果分成k组
			diffs_grouped = [[] for _ in range(k)]
			for label, cluster in kmeans_grouped.items():
				for item in cluster:	# item是各个客户端的rank序号, 0, 1, 2, ...
					client_dataset_items[label].append(split_idx[item])
					diffs_grouped[label].append(diffs[item])

			# 聚合k个全局模型参数
			for i in range(k):
				servers[i].model_aggregate(servers[i].calculate_weight_accumulator(diffs_grouped[i], client_dataset_items[i]))
				acc, acc_5, loss = servers[i].model_eval(kmeans_grouped[i])
				# Append accuracy and loss for this epoch to the corresponding lists
				acc_list[i].append(acc)
				loss_list[i].append(loss)
				message = f"Global Model {i}: acc = {acc} , acc_5 = {acc_5} loss = {loss}"
				notify_user(message, push)
			notify_user("[Global Epoch] Epoch " + str(e + 1) + " done!", push)

			for key, values in kmeans_grouped.items():	# key是label, values是label组的客户端rank
				for value in values:	# value是客户端rank
					if value != 0:
						# 将server[i]的全局模型的参数发送到客户端{value}
						for param, data in servers[key].global_model.state_dict().items():
							dist.send(data, dst=value)
					else:
						# 更新rank 0的客户端数据, 同步所属集群的全局模型的参数
						client.update_model(servers[key].global_model.state_dict())
		else:
			model_params_copy = copy.deepcopy(model_params)
			# 接收服务器发来的参数
			for key, value in model_params.items():	# 更新diff的值
				temp = torch.zeros_like(model_params[key], device=device)
				dist.recv(temp, src=0)
				model_params_copy[key] = temp

			# 更新本地模型的值
			client.update_model(model_params_copy)

		dist.barrier()

	# 只需要服务器去处理保存模型和保存accuracy/loss随epoch次数的变化图的操作
	if is_global_main_process():
		# 保存模型
		type = conf["type"]
		model_name = conf["model_name"]
		path = f"./result/{type}/{model_name}/"
		if not os.path.isdir(path):
			os.makedirs(path)
		for i in range(k):
			torch.save(servers[i].global_model.state_dict(), f"{path}{type}-{model_name}-{i}.pth")
			
		# 推送k个全局模型accuracy
		message = f"[Result Global Model] {type}-{model_name}-{k} accuracy = {acc_list}"
		notify_user(message, push)

		# 推送k个全局模型loss
		message = f"[Result Global Model] {type}-{model_name}-{k} loss = {loss_list}"
		notify_user(message, push)

		# 推送loss
		message = f"[Result Client] {type}-{model_name}-client-{k} loss = {client_loss_list}"
		notify_user(message, push)
		
		# 绘制准确率图像
		plt.clf()
		colors = get_colors_array(global_world_size)
		path = f"./figures/{type}/{model_name}/"
		if not os.path.isdir(path):
			os.makedirs(path)
		i = 0
		for acc in acc_list:
			plt.plot(range(1, conf["global_epochs"] + 1), acc, label=f'Global Model {i} Accuracy', linewidth=1.0, color=next(colors))
			i += 1
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
		plt.legend()
		plt.savefig(f"{path}{type}-{model_name}-global-accuracy.png")
		plt.show()

		# 绘制误差图像
		plt.clf()
		i = 0
		for loss in loss_list:
			plt.plot(range(1, conf["global_epochs"] + 1), loss, label=f'Global Model {i} Loss', linewidth=1.0, color=next(colors))
			i += 1
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig(f"{path}{type}-{model_name}-global-loss.png")
		plt.show()

		# 绘制各个客户端的accuracy图像
		# plt.clf()
		# i = 0
		# for acc in client_acc_list:
		# 	plt.plot(range(1, conf["global_epochs"] + 1), acc, label=f"Client{i} Accuracy", linewidth=1.0, color=next(colors))
		# 	i += 1
		# plt.xlabel('Epoch')
		# plt.ylabel('Accuracy')
		# plt.legend()
		# plt.savefig(f"{path}{type}-{model_name}-client-accuracy.png")
		# plt.show()

		# 绘制各个客户端的loss图像
		plt.clf()
		i = 0
		for loss in client_loss_list:
			plt.plot(range(1, conf["global_epochs"] + 1), loss, label=f"Client{i} Loss", linewidth=1.0, color=next(colors))
			i += 1
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig(f"{path}{type}-{model_name}-client-loss.png")
		plt.show()

	# Shutdown
	dist.destroy_process_group()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', type=str, default='./utils/conf.json', dest='conf')
	parser.add_argument('--gpu', type=int, default=-1, help='Which GPU to run,-1 means CPU, 0,1,2,... for GPU')
	args  = parser.parse_args()

	with open(args.conf, 'r') as f:
		conf = json.load(f)

	main(conf, args)