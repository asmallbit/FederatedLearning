import argparse, json
import datetime
import os
import logging
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

# 分别用来存储每次epoch的acc和loss
acc_list = []
loss_list = []

def main(conf, args):
	init_distributed_mode() # 进程组初始化
	local_rank = int(os.environ["LOCAL_RANK"])
	global_rank = int(os.environ["RANK"])
	local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
	gpu = args.gpu

	push = Push(conf)		# Push消息

	if local_world_size != 1:
		device = torch.device("cuda:{}".format(local_rank))
	elif gpu == 0:
		device = torch.device("cuda:{0}") # 只有1个GPU
	else:
		device = torch.device("cpu")

	if str(device) == "cpu":
		notify_user("[Client " + str(global_rank) + "] GPU is not enabled", push)
	else:
		notify_user("[Client " + str(global_rank) + "] GPU is using GPU " + str(local_rank) + " now", push)


	torch.manual_seed(5) # 方便复现
	train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
	global_model = models.get_model(conf["model_name"], True, device) # Set the flag to True to get pretrained model
	
	if is_global_main_process(global_rank):	# 服务端推送消息
		server = Server(conf, global_model, eval_datasets, push, device)
		message = "************************  TASK STARTED  ************************\nModel: " \
					+ conf["model_name"] + "\nDataset: " + conf["type"] \
					+ "\nBatch Size: " + str(conf["batch_size"]) + "\nNumber of Clients: " + os.environ["WORLD_SIZE"] \
					+ "\nLearing rate: " + str(conf["lr"]) + "\nMomentum: " + str(conf["momentum"]) \
					+ "\nGlobal Epochs: " + str(conf["global_epochs"]) + "\nLocal Epochs: " + str(conf["local_epochs"]) 

		notify_user(message, push)
		print("\n")
	
	client = Client(conf, global_model, train_datasets, push, global_rank, device)
	dist.barrier() # 确保客户端全部创建完毕

	for e in range(conf["global_epochs"]):		# global epochs 全局轮次
		if is_global_main_process(global_rank):
			# 每一轮epoch开始时提示
			message = "==============  Global Epoch " + str(e) + "  =============="
			notify_user(message, push)
			message = "[Global Model] Ephch " + str(e) + " started"
			notify_user(message, push)
		elif is_global_main_process(global_rank) and local_rank == 0:
			# 在local_rank的非根节点的terminal打印消息, 否则客户端的terminal并不能看到是第几轮训练
			message = "==============  Global Epoch " + str(e) + "  =============="
			print(message)
			message = "[Global Model] Ephch " + str(e) + " started"
			print(message)
		
		weight_accumulator = {}		# 权重累计, 初始化空模型参数weight_accumulator

		for name, params in global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params) # 生成一个和参数矩阵大小相同的0矩阵

		diff = client.local_train(global_model)	# 本地训练, 传入的全局模型并不参与运算, 只是用来求diff

		for name, params in global_model.state_dict().items():
			weight_accumulator[name].add_(diff[name])
				
		if is_global_main_process(global_rank):
			# 模型参数聚合
			notify_user("[Global Model] Aggregating the global model from trained local models", push)
			server.model_aggregate(weight_accumulator)
			
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
		dist.barrier()	# 保持epoch同步

	# 只需要服务器去处理保存模型和保存accuracy/loss随epoch次数的变化图的操作
	if is_global_main_process(global_rank):
		# 保存模型
		type = conf["type"]
		model_name = conf["model_name"]
		path = "./result/" + type + "/" + model_name + "/"
		if not os.path.isdir(path):
			os.makedirs(path)
		torch.save(server.global_model.state_dict(), path + type + "-" + model_name + ".pth")
		
		# 绘制准确率图像
		path = "./figures/" + type + "/" + model_name + "/"
		if not os.path.isdir(path):
			os.makedirs(path)
		plt.plot(range(conf["global_epochs"]), acc_list, label='Accuracy')
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
		plt.legend()
		plt.savefig(path + type + "-" + model_name + "-accuracy.png")
		plt.show()

		# 绘制误差图像
		plt.clf()
		plt.plot(range(conf["global_epochs"]), loss_list, label='Loss')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig(path + type + "-" + model_name + "-loss.png")
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
