import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import AttnEncoder

# one iteration of training
def train(X, ratings, attn_encoder, attn_encoder_optimizer):
	# X - single batch

	attn_encoder_optimizer.zero_grad()

	encoderLoss = nn.MSELoss(reduce=False)

	_, ratings_pred = attn_encoder(X)

	cost = encoderLoss(ratings_pred, ratings)

	cost_mean = cost.mean()

	scalar_cost = float(cost_mean)

	cost_mean.backward()

	return scalar_cost


attn_encoder = AttnEncoder.AttnEncoder(np.array([[1],[2]]), 2, 1, 3)