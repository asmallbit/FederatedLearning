import argparse
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
	server = Server(conf, global_model, eval_datasets, push, device)

	dataset_split_idx = None
	dataset_split_idx_size = 0
	split_idx = server.split_data()

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

	diffs = [dict() for _ in range(global_world_size)]

	if is_global_main_process():
		# 分别用来存储全局epoch的acc和loss
		acc_list = []
		loss_list = []
		# 分别用来存储每个客户端本地epoch的acc和loss
		# client_acc_list = [[] for _ in range(global_world_size)]
		client_loss_list = [[] for _ in range(global_world_size)]
		# 服务端推送消息
		message = (f"************************  TASK STARTED  ************************\nModel: {conf['model_name']}\n"
				f"Dataset: {conf['type']}\nBatch Size: {conf['batch_size']}\nNumber of Clients: {global_world_size}\n"
				f"Learning rate: {conf['lr']}\nGlobal Epochs: {conf['global_epochs']}\n"
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
						tensor = torch.tensor([0.0, 0.0])
						dist.recv(tensor, src=clinet_index)
						# client_acc_list/client_loss_client的长度为e
						# client_acc_list[clinet_index].append(tensor[0].item())	# 暂时不统计本地训练accuracy
						client_loss_list[clinet_index].append(tensor[1].item())

			# client_acc_list[0].append(0)
			client_loss_list[0].append(client_loss)
		else:
			tensor = torch.tensor([0, client_loss])
			dist.send(tensor, dst=0)
		dist.barrier()

		# 客户端向服务端发送本地训练后的模型和全局模型的diff, 服务端接收客户端发送的数据并保存
		if is_global_main_process():
			diffs[0] = model_params

			# 收集各个客户端的diff
			if global_world_size > 1:
				for key, _ in server.global_model.state_dict().items():
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
			server.model_aggregate(server.calculate_weight_accumulator(diffs, split_idx))
			acc, acc_5, loss = server.model_eval()
			# Append accuracy and loss for this epoch to the corresponding lists
			acc_list.append(acc)
			loss_list.append(loss)
			message = f"Global Model: acc = {acc} , acc_5 = {acc_5} loss = {loss}"
			notify_user(message, push)
			notify_user("[Global Epoch] Epoch " + str(e + 1) + " done!", push)

			for i in range(1, global_world_size):
				for key, value in server.global_model.state_dict().items():
					dist.send(value, dst=i)

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
		torch.save(server.global_model.state_dict(), path + type + "-" + model_name + ".pth")
		
		# 绘制准确率图像
		plt.clf()
		colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
		path = "./figures/" + type + "/" + model_name + "/"
		if not os.path.isdir(path):
			os.makedirs(path)
		plt.plot(range(1, conf["global_epochs"] + 1), acc_list, label=f'Global Model Accuracy', linewidth=1.0, color=colors[i % len(colors)])
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
		plt.legend()
		plt.savefig(path + type + "-" + model_name + "-global-accuracy.png")
		plt.show()

		# 绘制误差图像
		plt.clf()
		plt.plot(range(1, conf["global_epochs"] + 1), loss_list, label=f'Global Model Loss', linewidth=1.0, color=colors[i % len(colors)])
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig(path + type + "-" + model_name + "-global-loss.png")
		plt.show()

		# 绘制各个客户端的accuracy图像
		# plt.clf()
		# i = 0
		# for acc in client_acc_list:
		# 	plt.plot(range(1, conf["global_epochs"] + 1), acc, label=f"Client{i} Accuracy", linewidth=1.0, color=colors[i % len(colors)])
		# 	i += 1
		# plt.xlabel('Epoch')
		# plt.ylabel('Accuracy')
		# plt.legend()
		# plt.savefig(path + type + "-" + model_name + "-client-accuracy.png")
		# plt.show()

		# 绘制各个客户端的loss图像
		plt.clf()
		i = 0
		for loss in client_loss_list:
			plt.plot(range(1, conf["global_epochs"] + 1), loss, label=f"Client{i} Loss", linewidth=1.0, color=colors[i % len(colors)])
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
