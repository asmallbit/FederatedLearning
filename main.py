import argparse, json
import datetime
import os
import logging
import torch, random
import matplotlib.pyplot as plt

from server import *
from client import *
import models, datasets

# 分别用来存储每次epoch的acc和loss
acc_list = []
loss_list = []

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', default='utils/conf.json', dest='conf')
	args = parser.parse_args()
	

	with open(args.conf, 'r') as f:
		conf = json.load(f)	
	
	
	train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
	
	server = Server(conf, eval_datasets)
	clients = []
	
	for c in range(conf["no_models"]):
		clients.append(Client(conf, server.global_model, train_datasets, c))	# 创建client, 并将创建的client添加到clients
		
	print("\n\n")
	for e in range(conf["global_epochs"]):		# global epochs 全局轮次
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
		server.model_aggregate(weight_accumulator)
		
		acc, loss = server.model_eval()

		# Append accuracy and loss for this epoch to the corresponding lists
		acc_list.append(acc)
		loss_list.append(loss)
		
		print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

	# 保存模型
	type = conf["type"]
	model_name = conf["model_name"]
	path = "./result/" + type + "/" + model_name + "/"
	if not os.path.isdir(path):
		os.makedirs(path)
	torch.save(model.state_dict(), path + type + "-" + model_name + ".pth")
	
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
				
			
		
		
	
		
		
	