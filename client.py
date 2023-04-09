import models, torch, copy
from torch.utils.data import DataLoader, SubsetRandomSampler

from utils.output_handler import *

class Client(object):

	def __init__(self, conf, model, train_dataset, push: Push, id = -1):
		
		self.conf = conf
		
		self.local_model = models.get_model(self.conf["model_name"]) 
		
		self.client_id = id

		self.push = push
		
		self.train_dataset = train_dataset
		
		all_range = list(range(len(self.train_dataset)))
		data_len = int(len(self.train_dataset) / self.conf['no_models'])
		train_indices = all_range[id * data_len: (id + 1) * data_len]

		self.train_loader = DataLoader(self.train_dataset, batch_size=self.conf["batch_size"], 
                                    sampler=SubsetRandomSampler(train_indices), 
									num_workers=4, prefetch_factor=2)
									
	def local_train(self, model):

		for name, param in model.state_dict().items():
			self.local_model.state_dict()[name].copy_(param.clone())
	
		#print(id(model))
		optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])
		#print(id(self.local_model))
		self.local_model.train()

		message = "[Client] Client " + str(self.client_id) + " local train started"
		notify_user(message, self.push)
		
		for e in range(self.conf["local_epochs"]):
			message = "[Client] Client " + str(self.client_id) + " epoch " + str(e) + " started."
			notify_user(message, self.push)
			
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
			
				optimizer.zero_grad()
				output = self.local_model(data)
				loss = torch.nn.functional.cross_entropy(output, target)
				loss.backward()
			
				optimizer.step()
			message = "[Client] Client " + str(self.client_id) + " epoch " + str(e) + " done."
			notify_user(message, self.push)
		diff = dict()
		for name, data in self.local_model.state_dict().items():
			diff[name] = (data - model.state_dict()[name])
			#print(diff[name])
		message = "[Client] Client " + str(self.client_id) + " local train done"
		notify_user(message, self.push)

		return diff