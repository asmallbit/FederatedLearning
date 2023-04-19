import models, torch
from typing import Union

from push.push import Push

class Server(object):
	
	def __init__(self, conf, model, eval_dataset, push: Push, 
					device: Union[int, str]="cpu"):
	
		self.conf = conf 
		
		self.global_model = model

		self.device = device

		self.push = push
		
		self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=False)
		
	
	def model_aggregate(self, weight_accumulator):
		for name, data in self.global_model.state_dict().items():
			
			update_per_layer = weight_accumulator[name] * self.conf["lambda"]
			data = data.float()
			update_per_layer = update_per_layer.float()
			data.add_(update_per_layer)
				
	def model_eval(self):
		self.global_model.eval()
		
		total_loss = 0.0
		correct = 0
		dataset_size = 0
		for batch_id, batch in enumerate(self.eval_loader):
			data, target = batch
			data = data.to(self.device)
			target = target.to(self.device)

			dataset_size += data.size()[0]

			output = self.global_model(data)
			
			total_loss += torch.nn.functional.cross_entropy(output, target,
											  reduction='sum').item() # sum up batch loss
			pred = output.data.max(1)[1]  # get the index of the max log-probability
			correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

		acc = float(correct) / float(dataset_size)
		loss = total_loss / dataset_size

		return acc, loss