
import torch
from model.cat_and_dog import CatAndDogConvNet
from model.mnist import MNIST_CNN, MNIST_RNN
from torchvision import models
from typing import Union

def get_model(name="vgg16", pretrained=True, 
				device: Union[int, str]="cpu"): #默认为cpu
	if name == "resnet18":
		model = models.resnet18(pretrained=pretrained)
	elif name == "resnet50":
		model = models.resnet50(pretrained=pretrained)	
	elif name == "densenet121":
		model = models.densenet121(pretrained=pretrained)		
	elif name == "alexnet":
		model = models.alexnet(pretrained=pretrained)
	elif name == "vgg16":
		model = models.vgg16(pretrained=pretrained)
	elif name == "vgg19":
		model = models.vgg19(pretrained=pretrained)
	elif name == "inception_v3":
		model = models.inception_v3(pretrained=pretrained)
	elif name == "googlenet":		
		model = models.googlenet(pretrained=pretrained)
	elif name == "cat-and-dog-conv-net":
		model = CatAndDogConvNet()
	elif name == "mnist-cnn":
		model = MNIST_CNN()
	elif name == "mnist-rnn":
		model = MNIST_RNN()
	else:
		raise Exception("We don't support this model now")
	
	return model.to(device)