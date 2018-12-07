import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AttnEncoder(nn.Module):
	def __init__(self, pretrained_embeddings, rnn_hidden_dim, rnn_num_layers, linear1_hidden_dim, cell_type, dropout=0):
		super(AttnEncoder, self).__init__()

		vocab_size, input_dim = pretrained_embeddings.shape
		self.embed = nn.Embedding(vocab_size, input_dim)
		self.embed.weight.data = torch.from_numpy(pretrained_embeddings)
		self.embed.weight.requires_grad = False

		self.rnn_hidden_dim = rnn_hidden_dim
		self.rnn_num_layers = rnn_num_layers
		self.linear1_hidden_dim = linear1_hidden_dim
		self.cell_type = cell_type

		# self.lstm_i2h = nn.LSTM(input_dim, rnn_hidden_dim, rnn_num_layers)
		if (self.cell_type == 'LSTM'):
			self.lstm_i2h = nn.LSTM(input_dim, rnn_hidden_dim, rnn_num_layers)
		elif (self.cell_type == 'RNN'):
			self.rnn_i2h = nn.RNN(input_dim, rnn_hidden_dim, rnn_num_layers)

		self.linear1 = nn.Linear(2 * rnn_hidden_dim, linear1_hidden_dim)

		self.linear2 = nn.Linear(linear1_hidden_dim, 1)

		self.linear_final = nn.Linear(2 * rnn_hidden_dim, 1)

		self.dropout_init = nn.Dropout(dropout)
		self.dropout_final = nn.Dropout(dropout)

	def forward(self, x_index, use_cuda, train=True):
		x = self.embed(x_index)

		if (train):
			x = self.dropout_init(x)

		batch_size, seq_len = x_index.shape
		# x.shape - (batch, length, input_dim)

		x_transpose = torch.transpose(x, 1, 0)

		# hidden , (_, _) = self.lstm_i2h(x_transpose)

		if (self.cell_type == 'LSTM'):
			hidden, (h_n, c_n) = self.lstm_i2h(x_transpose)
			h_N = c_n[-1]
		elif (self.cell_type == 'RNN'):
			hidden, h_n = self.rnn_i2h(x_transpose)
			h_N = h_n[-1]

		# final hidden vector - dim - (batch, hidden_dim)

		# for i in range(seq_len):
		# 	attn_energies[i] = self.score(h_N, hidden[i])

		attn_energies = self.score( h_N.repeat(seq_len, 1, 1) , hidden )

		attn_energies = torch.transpose(attn_energies, 1, 0)
		attn_weights = F.softmax(attn_energies, dim=1)

		context_vec = torch.bmm(attn_weights.unsqueeze(1), torch.transpose(hidden, 1, 0)).squeeze(1)

		cat_vec = torch.cat((context_vec, h_N), dim=1)

		if (train):
			cat_vec = self.dropout_final(cat_vec)

		out = self.linear_final(cat_vec)

		return attn_weights, out.squeeze(1)


	def score(self, final_hidden, encoder_output):
		hidden_enc_out = torch.cat((final_hidden, encoder_output), dim=2)
		out_1 = F.tanh(self.linear1(hidden_enc_out))
		out_2 = self.linear2(out_1)
		return out_2.squeeze(2)



