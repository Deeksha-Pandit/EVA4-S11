import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from albumentations import Compose, Normalize, HorizontalFlip, Cutout, Rotate,  RandomCrop, FlipLR
from albumentations.pytorch import ToTensor
class album_Compose:	
	
	def __init__(self, train=False):
		self.train=train
		self.albumentations_transform_train = Compose([
												RandomCrop(height=32, width=32, always_apply=True),
												HorizontalFlip(),
												FlipLR(p=0.5),
												Normalize(
													mean=[0.5,0.5,0.5],
													std=[0.5,0.5,0.5]
												),
												Cutout(num_holes=1, max_h_size=8,max_w_size = 8,p = 1), # fillvalue is 0 after normalizing as mean is 0
												ToTensor()   
		])
		self.albumentations_transform_test = Compose([
												Normalize(
													mean=[0.5,0.5,0.5],
													std=[0.5,0.5,0.5]
												),
												ToTensor()   
		])
	
	def __call__(self,img):
		img = np.array(img)
		if self.train:
			img = self.albumentations_transform_train(image = img)['image']
		else:
			img = self.albumentations_transform_test(image = img)['image']
		return img
	
	"""transform = transforms.Compose(
		[transforms.RandomRotation(20),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])"""
	def load():
		SEED = 1


		# CUDA?
		cuda = torch.cuda.is_available()
		print("CUDA Available?", cuda)

		# For reproducibility
		torch.manual_seed(SEED)

		if cuda:
			torch.cuda.manual_seed(SEED)

		Batch_size = 64
		if cuda:
			Batch_size = 512
		

		# dataloader arguments - something you'll fetch these from cmdprmt
		dataloader_args = dict(shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=2)
		train_compose = album_Compose(train=True)
		test_compose = album_Compose()

		trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
												download=True, transform=train_compose)
		trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)

		testset = torchvision.datasets.CIFAR10(root='./data', train=False,
											download=True, transform=test_compose)
		testloader = torch.utils.data.DataLoader(testset, **dataloader_args)

		classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
		return classes, trainloader, testloader
