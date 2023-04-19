

import torch 
from torchvision import datasets, transforms
from dataset import dog_and_cat

def get_dataset(dir, name):
	
	if name=='mnist':
		train_transform = transforms.Compose([
			transforms.RandomRotation(degrees=15),	# 随即旋转
			transforms.RandomCrop(28, padding=2),	# 随机裁剪
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))	# 归一化
		])

		test_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
		])

		train_dataset = datasets.MNIST(f'{dir}/mnist', train=True, download=True, transform=train_transform)
		eval_dataset = datasets.MNIST(f'{dir}/mnist', train=False, download=True, transform=test_transform)
		
	elif name=='cifar':
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),	# 随机裁剪
			transforms.RandomHorizontalFlip(),		# 随机翻转
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		
		train_dataset = datasets.CIFAR10(f'{dir}/cifar-10', train=True, download=True,
										transform=transform_train)
		eval_dataset = datasets.CIFAR10(f'{dir}/cifar-10', train=False, transform=transform_test)
		
	elif name == 'dog-and-cat':
		transform_train = transforms.Compose([
			transforms.RandomResizedCrop(224),		# 随机裁剪224*224
			transforms.RandomHorizontalFlip(),		# 随机水平翻转
			transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),	# 随机改变颜色
			transforms.RandomRotation(degrees=15),	# 随机旋转图像
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),		# 
		])

		transform_test = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		])

		train_dataset = dog_and_cat.DogCatDataset(f'{dir}/dog-and-cat', train=True, download=True,
										transform=transform_train)
		eval_dataset = dog_and_cat.DogCatDataset(f'{dir}/dog-and-cat', train=False, transform=transform_test)

	return train_dataset, eval_dataset