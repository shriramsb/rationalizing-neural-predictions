import torch
import numpy as np
from torch import optim
import torch.nn as nn
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import random

import Generator
import Encoder
# Global defs
num_samples = 20
batch_size = 128

# iters_per_epoch should also be shifted here ?

# Function defs
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def asMinutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)

def timeSince(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

# Train function - here's some ingenuity
# one iteration of training
def train(X, ratings, encoder, generator, encoder_optimizer, generator_optimizer, length_reg, continuity_reg):
	# X - single batch

	encoder_optimizer.zero_grad()
	generator_optimizer.zero_grad()

	encoderLoss = nn.MSELoss(reduce=False)

	mean_cost = 0.0
	for i in range(num_samples):
		init_hidden = generator.initHidden()
		z_sample = generator.sample(X, init_hidden)
		z_sample = z_sample.detach()
		
		ratings_pred = encoder(X, z)
		encoder_loss = encoderLoss(ratings_pred, ratings)
		
		init_hidden = generator.initHidden()
		length_cost, continuity_cost = generator.cost(z_sample)

		cost = encoder_loss + length_cost * length_reg + continuity_cost * continuity_reg
		mean_cost += torch.mean(cost)

		log_prob = generator.logProb(X, z_sample, init_hidden)

		log_prob.backward(1.0 / (num_samples * batch_size) * cost)
		cost.backward(1.0 / (batch_size * num_samples))

	encoder_optimizer.step()
	generator_optimizer.step()

	mean_cost /= num_samples
	return mean_cost

def trainIters(X, ratings, encoder, generator, learning_rate, learning_rate_decay, num_epochs, length_reg, continuity_reg, print_every=1000, plot_every=100):

	plot_losses = []
	print_loss_total = 0.0
	plot_loss_total = 0.0

	encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
	generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)

	# set iters_per_epoch
	iters_per_epoch = num_train_examples / batch_size
	n_iters = iters_per_epoch * num_epochs

	encoder_scheduler = optim.lr_scheduler.stepLR(encoder_optimizer, 1, learning_rate_decay)
	generator_scheduler = optim.lr_scheduler.stepLR(generator_optimizer, 1, learning_rate_decay)

	for epoch in range(num_epochs):
		encoder_scheduler.step()
		generator_scheduler.step()
		for iter_num in range(iters_per_epoch):
			# randomly choose sample from X and make them equal length

			# This sampling also preserves the order
			X_bch = []
			ratings_bch = []

			_ = [ ( X_bch.append(X[i]) , ratings_bch.append(ratings[i]) ) for i in sorted(random.sample(range(num_train_examples), batch_size)) ]

			# almost done here - make all the reviews of equal length now

			maxlen_rev = max(X_bch, key=len)
			maxlen = len(maxlen_rev)

			X_bach = np.empty([0,maxlen])
			ratings_bach = np.empty([0,1])

			for iterr in range(batch_size):
				currentlen = len(X_bch[iterr])
				zero_count = maxlen - currentlen
				X_bch[iterr].extend([0]*zero_count)
				# X_bch[iterr] is now a list containing indices of words
				# Convert it into a Variable ?
				to_append = np.array( X_bch[iterr] )
				X_bach = np.append(X_bach, [to_append], axis = 0)
				to_append = np.array( ratings_bch[iterr] )
				ratings_bach = np.append(ratings_bach, to_append, axis = 0)
			# X_bach is a 2d numpy array of size :: batch_size X maxlen

			X_bach_tensor = torch.from_numpy(X_bach)
			X_batch = Variable(X_bach_tensor)
			ratings_bach_tensor = torch.from_numpy(ratings_bach)
			ratings_batch = Variable(ratings_bach_tensor)
			# call train with this batch
			cur_loss = train(X_batch, ratings_batch, encoder, generator, encoder_optimizer, generator_optimizer, length_reg, continuity_reg)

			print_loss_total += cur_loss
			plot_loss_total += cur_loss

			if iter_num % print_every == 0:
				print_loss_avg = print_loss_total / print_every
				print_loss_total = 0
				print('%s (%d %d%%) %.4f' % (timeSince(start, (iter_num + epoch * iters_per_epoch) / n_iters),
											 iter_num + epoch * iters_per_epoch, (iter_num + epoch * iters_per_epoch) / n_iters * 100, print_loss_avg))

				getAccuracy(X_val, ratings_val, encoder, generator)

			if iter_num % plot_every == 0:
				plot_loss_avg = plot_loss_total / plot_every
				plot_losses.append(plot_loss_avg)
				plot_loss_total = 0

	showPlot(plot_losses)


def getAccuracy(X, ratings, encoder, generator):

	# iterate through X_val and pass to generator->encoder to get mse_error and compare it to truth
	X_val_size = num_val_examples
	num_iters = X_val_size / (1.0*batch_size)
	total_loss = 0.0
	for iters in range(num_iters):
		
		# get X_batch, ratings_batch
		# This sampling also preserves the order
		X_bch = []
		ratings_bch = []

		_ = [ ( X_bch.append(X[i]) , ratings_bch.append(ratings[i]) ) for i in sorted(random.sample(range(num_train_examples), batch_size)) ]

		# almost done here - make all the reviews of equal length now

		maxlen_rev = max(X_bch, key=len)
		maxlen = len(maxlen_rev)

		X_bach = np.empty([0,maxlen])
		ratings_bach = np.empty([0,1])

		for iterr in range(batch_size):
			currentlen = len(X_bch[iterr])
			zero_count = maxlen - currentlen
			X_bch[iterr].extend([0]*zero_count)
			# X_bch[iterr] is now a list containing indices of words
			# Convert it into a Variable ?
			to_append = np.array( X_bch[iterr] )
			X_bach = np.append(X_bach, [to_append], axis = 0)
			to_append = np.array( ratings_bch[iterr] )
			ratings_bach = np.append(ratings_bach, to_append, axis = 0)
		# X_bach is a 2d numpy array of size :: batch_size X maxlen

		X_bach_tensor = torch.from_numpy(X_bach)
		X_batch = Variable(X_bach_tensor)
		ratings_bach_tensor = torch.from_numpy(ratings_bach)
		ratings_batch = Variable(ratings_bach_tensor)

		init_hidden = generator.initHidden()
		z_sample = generator.sample(X_batch, init_hidden)
		
		ratings_pred = encoder(X_batch, z_sample)
		encoderLoss = nn.MSELoss(reduce=False)
		encoder_loss = encoderLoss(ratings_pred, ratings_batch)
		
		total_loss += torch.mean(encoder_loss)

	return total_loss / X_val_size

# ** Starting of the Main code 
# ** ** 
# ** ** 
import re

# Convert string to vector of floats
def convert_to_float(string): # string with float values separated by spaces
	lis = string.split()
	lis_rating = [ float(value) for value in lis]
	return lis_rating

# Unique index for words
index = 0
def get_index():
	global index
	to_ret = index
	index += 1
	return to_ret

# Dictionaries
dict_ind2vec = {}
dict_ind2str = {}
dict_str2ind = {}

def get_list_of_indices(string):
	lis_words = string.split()
	# lis_ret = [ for word in lis_words]
	lis_ret = []
	for word in lis_words:
		try:
			ind_append = dict_str2ind[word]
			lis_ret.append(ind_append)
		except:
			pass
			# ind_append = 
			# print("THERE IT IS!", word)
	# print("About to return")
	return lis_ret

## **
## **
# read the word2vec representations

with open('../review+wiki.filtered.200.txt') as f:
	wordvecs = f.readlines()

first_pair = wordvecs[0].split(" ", 1)
first_vec = convert_to_float(first_pair[1])
dim_vecSpace = len(first_vec) # Dimension of the vector space in which we are

# add stuff for EOS, Blank
# at index = 0, 1

eos_index = get_index()
dict_str2ind["<EOS>"] = eos_index
dict_ind2str[eos_index] = "<EOS>"
dict_ind2vec[eos_index] = [1.0]*dim_vecSpace

blk_index = get_index()
dict_str2ind["<BLANK>"] = blk_index
dict_ind2str[blk_index] = "<BLANK>"
dict_ind2vec[blk_index] = [0.0]*dim_vecSpace


for elem in wordvecs:
	liss = elem.split(" ", 1) # split on the first space
	word_str = liss[0]
	word_vec = convert_to_float(liss[1])
	
	here_index = get_index()
	dict_str2ind[word_str] = here_index
	dict_ind2str[here_index] = word_str
	dict_ind2vec[here_index] = word_vec

# CHKING
# print( dict_str2ind['a'] )

## **
## **
# read the data

with open('../reviews.aspect0.train.txt') as f:
	train_data = f.readlines()

rating_regex = re.compile('\d\.\d\d \d\.\d\d \d\.\d\d \d\.\d\d \d\.\d\d\t') # Exactly matches only the ratings

# extract ratings - # each rating is a scalar value # NO ::: each rating is a list of 5 values
ratings = [ float( re.findall(rating_regex, review)[0][0] ) for review in train_data ]

# extract reviews
reviews_str = [ rating_regex.sub('', review) for review in train_data ]
reviews = [ get_list_of_indices( review_str ) for review_str in reviews_str ]
X = reviews
total_size = len(X)

divide_train = int( (4*total_size)/5 )
train_indices_of_X = sorted( random.sample( range(total_size), divide_train ) )

X_train = []
X_val = []
ratings_train = []
ratings_val = []
for i in range(total_size):
	if i in train_indices_of_X:
		X_train.append(X[i])
		ratings_train.append(ratings[i])
	else:
		X_val.append(X[i])
		ratings_val.append(ratings[i])

X = X_train
ratings = ratings_train

num_train_examples = len(X) # we also assume len(X) = len(ratings)
num_val_examples = len(X_val)
# read validation data

# ** ** 
# ** ** 

# Initialing hyperparam containers
learning_rates = []
length_regs = []
continuity_regs = []
learning_rate_decays = []
num_epochs = 100
gen_hidden_dim = 200
gen_num_layers = 1
gen_s_size = 30



for lrate_decay in learning_rate_decays:
	for length_reg in length_regs:
		for continuity_reg in continuity_reg:
			for l_rate in learning_rates:
				generator = Generator.Generator(gen_hidden_dim, gen_num_layers, gen_s_size, pretrained_embeddings, 'LSTM')
				# fill encoder parameters
				encoder = Encoder.Encoder()

				trainIters(X, ratings, encoder, generator, 
							learning_rate=l_rate, learning_rate_decay=lrate_decay, num_epochs=num_epochs, 
							length_reg=length_reg, continuity_reg=continuity_reg)



