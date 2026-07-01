# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm1d(nn.Module):
	# TODO START
	def __init__(self, num_features, eps=1e-5, momentum=0.1):
		super(BatchNorm1d, self).__init__()
		self.num_features = num_features
		self.eps = eps
		self.momentum = momentum

		# Parameters
		self.weight = Parameter(torch.ones(num_features))
		self.bias = Parameter(torch.zeros(num_features))

		# Store the average mean and variance
		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))
		
		# Initialize your parameter
		# Weight and bias are already initialized above

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.training:
			mean = input.mean(dim=0)  # [num_features]
			var = input.var(dim=0, unbiased=False)  # [num_features]

			self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
			self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
		else:
			mean = self.running_mean
			var = self.running_var
		input_norm = (input - mean) / torch.sqrt(var + self.eps)
		output = self.weight * input_norm + self.bias
		return output
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.training and self.p > 0:
			mask = torch.rand_like(input) > self.p
			output = input * mask.float() / (1 - self.p)
			return output
		else:
			# During inference or when p=0, return input unchanged
			return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5):
		super(Model, self).__init__()
		# TODO START
		input_size = 3 * 32 * 32   # 3072
		hidden1 = 256
		feat_num = 10
		self.lin1 = nn.Linear(input_size, hidden1)
		self.lin2 = nn.Linear(hidden1, feat_num)
		self.batchn = BatchNorm1d(hidden1)
		self.dropout = Dropout(drop_rate)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		x = self.batchn(self.lin1(x))
		x = nn.functional.relu(x)
		logits = self.lin2(self.dropout(x))
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
