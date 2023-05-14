import models, torch, copy, os
import multiprocessing as mp
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from typing import Union

import torch.distributed as dist
from utils.distributed_utils import *
from utils.output_handler import *

class Client(object):

	def __init__(self, conf, model, train_dataset, eval_dataset, push: Push, dataset_split_idx,
					device: Union[int, str]="cpu"):
		
		self.conf = conf

		self.local_model = copy.deepcopy(model)
		
		self.client_id = get_global_rank()

		self.device = device

		self.push = push
		
		self.train_dataset = train_dataset

		self.eval_dataset = eval_dataset
		
		self.criterion = torch.nn.CrossEntropyLoss()

		self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr = self.conf['lr']) 

		self.train_loader = DataLoader(dataset=Subset(self.train_dataset, np.sort(dataset_split_idx)),
									shuffle=True,
									drop_last=True,
									batch_size=self.conf["batch_size"],
									num_workers = mp.cpu_count(),
									prefetch_factor=2, pin_memory=True) # 这里pin_memory的作用是加速GPU读取数据

		self.eval_loader = DataLoader(eval_dataset, batch_size=self.conf["batch_size"],
									shuffle=False,
									num_workers = mp.cpu_count(),
									prefetch_factor=2, pin_memory=True)

	def update_model(self, diff):
		self.local_model.load_state_dict(diff)
									
	def local_train(self):
		# space = [Real(0.0001, 0.1, name='learning_rate')]
		# for name, param in model.state_dict().items():
		# 	self.local_model.state_dict()[name].copy_(param.clone())
		# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
															# factor=self.conf['factor'], patience=self.conf['patience'])
		self.local_model.train()

		message = "[Client] Client " + str(self.client_id) + " local train started"
		notify_user(message, self.push)
		loss_avg = 0
		for e in range(self.conf["local_epochs"]):
			message = "[Client] Client " + str(self.client_id) + " epoch " + str(e + 1) + " started."
			notify_user(message, self.push)
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				data = data.to(self.device)
				target = target.to(self.device)
			
				output = self.local_model(data)
				loss = self.criterion(output, target)
				self.optimizer.zero_grad()
				loss.backward()
				loss_avg += loss.item() / (len(self.train_loader) * self.conf["local_epochs"])
				self.optimizer.step()

				if (batch_id + 1) % 100 == 0:
					notify_user("[Client {}] Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
								get_global_rank(), e + 1, self.conf["local_epochs"], batch_id + 1, 
								len(self.train_loader), loss),
							self.push)
				# 发送acc和loss到服务器(rank 0)
				if batch_id + 1 == len(self.train_loader) and e + 1 == self.conf["local_epochs"]:
					message = "[Client] Client " + str(self.client_id) + " epoch " + str(e + 1) + " done."
					notify_user(message, self.push)

					message = "[Client] Client " + str(self.client_id) + " local train done"
					notify_user(message, self.push)

					return self.local_model.state_dict(), loss_avg

		return dict(), 99999 # 仅起到占位作用， 实际上不会用到