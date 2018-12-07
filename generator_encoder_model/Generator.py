import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import Variable

class Generator(nn.Module):
	def __init__(self, hidden_dim, num_layers, s_size, pretrained_embeddings, cell_type):
		# pretrained_embeddings - numpy array
		super(Generator, self).__init__()
		vocab_size, input_dim = pretrained_embeddings.shape
		self.embed = nn.Embedding(vocab_size, hidden_dim)
		self.embed.weight.data = torch.from_numpy(pretrained_embeddings)
		self.embed.weight.requires_grad = False
		
		self.i2h_num_layers = num_layers
		self.i2h_hidden_dim = hidden_dim

		# bidirectional lstm 
		if (cell_type == 'LSTM'):
			self.lstm_i2h = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=True)

		# keep track of previous words chosen
		self.s_size = s_size
		self.lstm_h2s = nn.LSTM(2 * hidden_dim + 1, s_size, 1)
		self.h2o = nn.Linear(2 * hidden_dim + s_size, 1)

	def logProb(self, x_index, z, init_hidden, use_cuda):
		lstm_i2h_h0, lstm_i2h_c0, lstm_h2s_h0, lstm_h2s_c0 = init_hidden
		# dim lstm_h2s_h0 - (num_layers * num_directions, batch, hidden_size)
		# dim lstm_h2s_c0 - (num_layers * num_directions, batch, hidden_size)

		# x_index of type (batch, length of para)
		batch_size, seq_len = x_index.size()
		x = self.embed(x_index)
		# x of type (batch, length of para, embed length)

		x = torch.transpose(x, 1, 0)

		hidden_i2h, (_, _) = self.lstm_i2h(x, (lstm_i2h_h0, lstm_i2h_c0))

		z_transformed = torch.transpose(z, 1, 0)
		z_transformed_unsqueezed = torch.unsqueeze(z_transformed, 2)
		s_h2s, (_, _) = self.lstm_h2s(torch.cat((hidden_i2h, z_transformed_unsqueezed), dim=2), (lstm_h2s_h0, lstm_h2s_c0))


		if use_cuda:
			log_p_z = Variable(torch.zeros((batch_size)).cuda())
		else:
			log_p_z = Variable(torch.zeros((batch_size)))

		for i in range(seq_len):
			if (i == 0):
				cur_p_z = self.h2o(torch.cat((hidden_i2h[i], lstm_h2s_h0[0]), dim=1))
			else:
				cur_p_z = self.h2o(torch.cat((hidden_i2h[i], s_h2s[i - 1]), dim=1))

			cur_p_z = F.sigmoid(torch.squeeze(cur_p_z, 1))
			cur_p_z = z_transformed[i] * cur_p_z + (1 - z_transformed[i]) * (1 - cur_p_z)
			cur_log_p_z = torch.log(cur_p_z)

			log_p_z = log_p_z + cur_log_p_z

		return log_p_z


	def sample(self, x_index, init_hidden, use_cuda):
		lstm_i2h_h0, lstm_i2h_c0, lstm_h2s_h0, lstm_h2s_c0 = init_hidden
		batch_size, seq_len = x_index.size()

		x = torch.transpose(self.embed(x_index), 1, 0)

		hidden_i2h , (_, _) = self.lstm_i2h(x, (lstm_i2h_h0, lstm_i2h_c0))

		if use_cuda:
			z = torch.zeros((seq_len, batch_size)).cuda()
		else:
			z = torch.zeros((seq_len, batch_size))

		z = Variable(z)

		s_h2s_h = lstm_h2s_h0
		s_h2s_c = lstm_h2s_c0

		for i in range(seq_len):
			cur_p_z = self.h2o(torch.cat((hidden_i2h[i], s_h2s_h[0]), dim=1))

			cur_p_z = F.sigmoid(torch.squeeze(cur_p_z, 1))
			m = Bernoulli(cur_p_z)
			z[i] = m.sample()


			cat_hidden_z = torch.unsqueeze(torch.cat((hidden_i2h[i], torch.unsqueeze(z[i], 1)), dim=1), 0)
			_, (s_h2s_h, s_h2s_c) = self.lstm_h2s(cat_hidden_z, (s_h2s_h, s_h2s_c))


		return torch.transpose(z, 1, 0)

	def initHidden(self, batch_size, use_cuda):
		if use_cuda:
			lstm_i2h_h0 = Variable(torch.zeros(self.i2h_num_layers * 2, batch_size, self.i2h_hidden_dim).cuda())
			lstm_i2h_c0 = Variable(torch.zeros(self.i2h_num_layers * 2, batch_size, self.i2h_hidden_dim).cuda())

			lstm_h2s_h0 = Variable(torch.zeros(1 * 1, batch_size, self.s_size).cuda())
			lstm_h2s_c0 = Variable(torch.zeros(1 * 1, batch_size, self.s_size).cuda())

		else:
			lstm_i2h_h0 = Variable(torch.zeros(self.i2h_num_layers * 2, batch_size, self.i2h_hidden_dim))
			lstm_i2h_c0 = Variable(torch.zeros(self.i2h_num_layers * 2, batch_size, self.i2h_hidden_dim))

			lstm_h2s_h0 = Variable(torch.zeros(1 * 1, batch_size, self.s_size))
			lstm_h2s_c0 = Variable(torch.zeros(1 * 1, batch_size, self.s_size))

		return lstm_i2h_h0, lstm_i2h_c0, lstm_h2s_h0, lstm_h2s_c0

	def loss(self, z):
		length_cost = torch.sum(z, dim=1)
		l_padded_mask =  torch.cat([z[:,0].unsqueeze(1), z] , dim=1)
		r_padded_mask =  torch.cat([z, z[:,-1].unsqueeze(1)] , dim=1)
		continuity_cost = torch.sum(torch.abs(r_padded_mask - l_padded_mask), dim=1)
		return length_cost, continuity_cost
