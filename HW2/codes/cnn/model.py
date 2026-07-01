# -*- coding: utf-8 -*-

from re import X
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm2d(nn.Module):
	# TODO START
	def __init__(self, num_features, eps=1e-5, momentum=0.1):
		super(BatchNorm2d, self).__init__()
		self.num_features = num_features
		self.eps = eps
		self.momentum = momentum
		self.weight = Parameter(torch.ones(num_features))
		self.bias = Parameter(torch.zeros(num_features))

		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))


	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		if self.training:
			mean = input.mean(dim=[0, 2, 3])  # [num_features]
			var = input.var(dim=[0, 2, 3], unbiased=False)  # [num_features]

			self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
			self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
		else:
			# Inference mode: use running statistics
			mean = self.running_mean
			var = self.running_var
		input_norm = (input - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
		output = self.weight.view(1, -1, 1, 1) * input_norm + self.bias.view(1, -1, 1, 1)
		
		return output
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		if self.training and self.p > 0:
			mask = torch.bernoulli(torch.full_like(input, 1 - self.p))
			output = input * mask / (1 - self.p)
			return output
		else:
			return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5):
		super(Model, self).__init__()
		# TODO START
		input_size = 32
		inchannels = 3
		hidden1 = 48
		hidden2 = 96
		kernel1 = 6
		kernel2 = 3
		pool_kernel = 2
		padding_ = 1
  
		self.conv1 = nn.Conv2d(inchannels, hidden1, kernel_size=kernel1, padding=padding_)
		self.batchn1 = BatchNorm2d(hidden1)
		self.pool1 = nn.MaxPool2d(kernel_size=pool_kernel)   
		self.conv2 = nn.Conv2d(hidden1, hidden2, kernel_size=kernel2, padding=padding_)
		self.batchn2 = BatchNorm2d(hidden2)
		self.pool2 = nn.MaxPool2d(kernel_size=pool_kernel)   
  
		conv1_size = (input_size - kernel1 + 2*padding_) // 1 + 1   # stride = 1
		conv1_size = (conv1_size - pool_kernel) // pool_kernel + 1
		conv2_size = (conv1_size - kernel2 + 2*padding_) // 1 + 1
		conv2_size = (conv1_size - pool_kernel) // pool_kernel + 1
		lin_input = hidden2 * conv2_size * conv2_size
  
		self.lin = nn.Linear(lin_input, 10)
		self.dropout = Dropout(drop_rate)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):	
		# TODO START
		x = self.conv1(x)  
		x = self.batchn1(x)
		x = nn.functional.relu(x)
		x = self.dropout(x)
		x = self.pool1(x)  
		
		x = self.conv2(x)  
		x = self.batchn2(x)
		x = nn.functional.relu(x)
		x = self.dropout(x)
		x = self.pool2(x)  
  
		x = x.view(x.size(0), -1)
		logits = self.lin(x) 
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
