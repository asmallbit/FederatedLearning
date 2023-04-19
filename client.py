import models, torch, copy, os
# from skopt.space import Real
import pickle
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import Union

import torch.distributed as dist
from utils.distributed_utils import *
from utils.output_handler import *

class Client(object):

	def __init__(self, conf, model, train_dataset, eval_dataset, push: Push,
					device: Union[int, str]="cpu"):
		
		self.conf = conf

		self.local_model = model
		
		self.client_id = get_global_rank()

		self.device = device

		self.push = push
		
		self.train_dataset = train_dataset

		self.eval_dataset = eval_dataset
		
		all_range = list(range(len(self.train_dataset)))
		data_len = int(len(self.train_dataset) / get_global_world_size())
		train_indices = all_range[self.client_id * data_len: (self.client_id + 1) * data_len]

		self.train_loader = DataLoader(self.train_dataset, batch_size=self.conf["batch_size"], 
                                    sampler=SubsetRandomSampler(train_indices), shuffle=True,
									num_workers=4, prefetch_factor=2, pin_memory=True) # 这里pin_memory的作用是加速GPU读取数据

		self.eval_loader = DataLoader(eval_dataset, batch_size=self.conf["batch_size"],
									shuffle=False,
									num_workers=4, prefetch_factor=2, pin_memory=True)
									
	def local_train(self, model):
		# space = [Real(0.0001, 0.1, name='learning_rate')]
		# for name, param in model.state_dict().items():
		# 	self.local_model.state_dict()[name].copy_(param.clone())

		self.local_model = copy.deepcopy(model)
	
		optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
															factor=self.conf['factor'], patience=self.conf['patience'])
		self.local_model.train()

		message = "[Client] Client " + str(self.client_id) + " local train started"
		notify_user(message, self.push)
		
		for e in range(self.conf["local_epochs"]):
			message = "[Client] Client " + str(self.client_id) + " epoch " + str(e) + " started."
			notify_user(message, self.push)
			
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				data = data.to(self.device)
				target = target.to(self.device)
			
				optimizer.zero_grad()
				output = self.local_model(data)
				loss = torch.nn.functional.cross_entropy(output, target)
				loss.backward()
				optimizer.step()

				scheduler.step(loss)	# 调整学习率

			message = "[Client] Client " + str(self.client_id) + " epoch " + str(e) + " done."
			notify_user(message, self.push)

		diff = dict()
		for name, data in self.local_model.state_dict().items():
			diff[name] = (data - model.state_dict()[name])
			#print(diff[name])

		message = "[Client] Client " + str(self.client_id) + " local train done"
		notify_user(message, self.push)

		# 非rank 0的客户端进程通过send()方法和rank 0进程进行交互，rank 0的客户端进程通过返回值传递到main方法
		# if not is_global_main_process():
		# 	# 分别将长度和二进制的diff数据发送到server
		# 	diff_binary = pickle.dumps(diff)
		# 	len_tensor = torch.tensor([len(diff_binary)], dtype=torch.int)
		# 	dist.send(len_tensor, 0) # 发送长度
		# 	dist.send(diff_binary, 0)	# 发送二进制数据

		return diff


	def model_eval(self, epoch):
		self.local_model.eval()
		
		total_loss = 0.0
		correct = 0
		dataset_size = 0
		for batch_id, batch in enumerate(self.eval_loader):
			data, target = batch
			data = data.to(self.device)
			target = target.to(self.device)

			dataset_size += data.size()[0]

			output = self.local_model(data)
			
			total_loss += torch.nn.functional.cross_entropy(output, target,
											  reduction='sum').item() # sum up batch loss
			pred = output.data.max(1)[1]  # get the index of the max log-probability
			correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

		acc = float(correct) / float(dataset_size)
		loss = total_loss / dataset_size

		message = f'[Client {get_global_rank()}] Epoch {epoch}, acc = {acc}, loss = {loss}'
		notify_user(message, self.push)

		# 发送acc和loss到服务器(rank 0)
		if not is_global_main_process():
			tensor = torch.tensor([acc, loss])
			dist.send(tensor, 0)
		
		return acc, loss