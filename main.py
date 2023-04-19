import argparse, json
import datetime
import os
import logging
import pickle
import sys
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

# 分别用来存储每次epoch的acc和loss
acc_list = []
loss_list = []

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
	
	# TODO: server应该是应该只在主节点, 这里为了每个节点都能更新全局模型, 在所有参与训练的进程都部署一个server
	server = Server(conf, global_model, eval_datasets, push, device)

	if is_global_main_process():
		client_acc_list = [[] for _ in range(global_world_size)]
		client_loss_list = [[] for _ in range(global_world_size)]
		# 服务端推送消息
		# server = Server(conf, global_model, eval_datasets, push, device)

		message = (f"************************  TASK STARTED  ************************\nModel: {conf['model_name']}\n"
				f"Dataset: {conf['type']}\nBatch Size: {conf['batch_size']}\nNumber of Clients: {global_world_size}\n"
				f"Learning rate: {conf['lr']}\nMomentum: {conf['momentum']}\nFactor: {conf['factor']}\n"
				f"Patience: {conf['patience']}\nGlobal Epochs: {conf['global_epochs']}\n"
				f"Local Epochs: {conf['local_epochs']}")

		notify_user(message, push)
		print("\n")
	client = Client(conf, server.global_model, train_datasets, eval_datasets, push, device)
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

		diff = client.local_train(global_model)
		dist.barrier()	# 训练完毕

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

		# 聚合diff
		for key, value in diff.items():
			dist.all_reduce(value, op=dist.ReduceOp.SUM)
			dist.barrier()

		# TODO: server应该是应该只在主节点, 这里为了每个节点都能更新全局模型, 在所有参与训练的进程都部署一个server并进行更新
		server.model_aggregate(diff)

		if is_global_main_process():
			# 模型参数聚合
			notify_user("[Global Model] Aggregating the global model from trained local models", push)
			# server.model_aggregate(weight_accumulator)
			
			acc, loss = server.model_eval()

			# Append accuracy and loss for this epoch to the corresponding lists
			acc_list.append(acc)
			loss_list.append(loss)
			message = "[Global Epoch] Epoch " + str(e) + " done, acc = " + str(acc) + ", loss = " + str(loss)
			notify_user(message, push)
			# 设置终止条件
			if len(acc_list) > 1 and abs(acc - acc_list[-2]) < conf["accuracy_difference_threshold"] \
					and abs(loss - loss_list[-2] < conf["loss_difference_threshold"]):
				notify_user("Model has converged, stop training", push)
				break
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
		colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
		path = "./figures/" + type + "/" + model_name + "/"
		if not os.path.isdir(path):
			os.makedirs(path)
		plt.plot(range(conf["global_epochs"]), acc_list, label='Global Accuracy', linewidth=1.0)
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
		plt.legend()
		plt.savefig(path + type + "-" + model_name + "-accuracy.png")
		plt.show()

		# 绘制误差图像
		plt.clf()
		plt.plot(range(conf["global_epochs"]), loss_list, label='Global Loss', linewidth=1.0)
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig(path + type + "-" + model_name + "-loss.png")
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
