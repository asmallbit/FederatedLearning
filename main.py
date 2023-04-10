import argparse, json
import datetime
import os
import logging
import torch, random
import matplotlib.pyplot as plt

from server import *
from client import *
import models, datasets
from push.push import Push
from utils.output_handler import *

# 分别用来存储每次epoch的acc和loss
acc_list = []
loss_list = []

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', default='utils/conf.json', dest='conf')
	args = parser.parse_args()
	

	with open(args.conf, 'r') as f:
		conf = json.load(f)
	
	push = Push(conf)		# push消息
	
	train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
	
	server = Server(conf, eval_datasets, push)
	clients = []

	message = "************************  TASK STARTED  ************************\nModel: " \
				+ conf["model_name"] + "\nDataset: " + conf["type"] \
				+ "\nBatch Size: " + str(conf["batch_size"]) + "\nNumber of Clients: " + str(conf["no_models"]) \
				+ "\nk: " + str(conf["k"]) \
				+ "\nLearing rate: " + str(conf["lr"]) + "\nMomentum: " + str(conf["momentum"]) \
				+ "\nGlobal Epochs: " + str(conf["global_epochs"]) + "\nLocal Epochs: " + str(conf["local_epochs"]) 

	notify_user(message, push)
	
	for c in range(conf["no_models"]):
		clients.append(Client(conf, server.global_model, train_datasets, push, id = c))	# 创建client, 并将创建的client添加到clients

	print("\n")

	for e in range(conf["global_epochs"]):		# global epochs 全局轮次
		# 每一轮epoch开始时提示
		message = "==============  Global Epoch " + str(e) + "  =============="
		notify_user(message, push)
		message = "[Global Model] Ephch " + str(e) + " started"
		notify_user(message, push)

		# 每次训练都是从clients列表中随机采样k个进行本轮训练
		candidates = random.sample(clients, conf["k"])		# 每轮参与的客户端数量
		
		weight_accumulator = {}		# 权重累计, 初始化空模型参数weight_accumulator
		
		for name, params in server.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params) # 生成一个和参数矩阵大小相同的0矩阵
		
		# 遍历客户端，每个客户端本地训练模型
		for c in candidates:
			diff = c.local_train(server.global_model)
			
			for name, params in server.global_model.state_dict().items():
				weight_accumulator[name].add_(diff[name])
				
		
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
	# plt.plot(range(conf["global_epochs"]), loss_list, label='Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.savefig(path + type + "-" + model_name + "-accuracy.png")
	plt.show()

	# 绘制误差图像
	plt.plot(range(conf["global_epochs"]), loss_list, label='Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig(path + type + "-" + model_name + "-loss.png")
	plt.show()
				
			
		
		
	
		
		
	