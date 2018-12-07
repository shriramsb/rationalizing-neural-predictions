import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Encoder(nn.Module):
	def __init__(self, hidden_dim, num_layers, pretrained_embeddings, cell_type, dropout=0.1):
	# pretrained_embeddings - numpy array
		super(Encoder, self).__init__()
		vocab_size, input_dim = pretrained_embeddings.shape
		# vocab_size, input_dim = (5, 1)
		self.embed = nn.Embedding(vocab_size, input_dim)
		self.embed.weight.data = torch.from_numpy(pretrained_embeddings)
		self.embed.weight.requires_grad = False

		self.cell_type = cell_type
		if (cell_type == 'LSTM'):
			self.lstm_i2h = nn.LSTM(input_dim, hidden_dim, num_layers)
		elif (cell_type == 'RNN'):
			self.rnn_i2h = nn.RNN(input_dim, hidden_dim, num_layers)

		self.h2o = nn.Linear(hidden_dim, 1)
		self.dropout_init = nn.Dropout(dropout)
		self.dropout_final = nn.Dropout(dropout)

	def forward(self, x_index, z, train=True):
		# x_index, z - variable

		x = self.embed(x_index)
		if (z is not None):
			x = x * z.unsqueeze(-1)

		if (train):
			x = self.dropout_init(x)

		# add dropout if required

		x = torch.transpose(x, 1, 0)

		# initial state always zero
		if (self.cell_type == 'LSTM'):
			output, (_, _) = self.lstm_i2h(x)
		elif (self.cell_type == 'RNN'):
			output, _ = self.rnn_i2h(x)

		if (train):
			last_output = self.dropout_final(output[-1])
		else:
			last_output = output[-1]

		# take the final hidden state and use it for prediction
		pred = self.h2o(last_output)

		# return predictions
		return pred.squeeze(1)
