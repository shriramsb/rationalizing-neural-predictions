import numpy as np

# softmax attn_weight clustering
def clusterAttnWeights(attn_weights):
	# attn_weights in 1d numpy array
	seq_len = attn_weights.shape[0]

	attn_weights = np.sort(attn_weights)
	
	mean_1 = 0
	mean_2 = attn_weights.mean()
	sse_1 = 0
	sse_2 = ((attn_weights - mean_2) ** 2).sum()

	best_sse = sse_1 + sse_2
	best_val = attn_weights[0] - 1
	
	N = seq_len
	for i in range(seq_len - 1):
		x_i = attn_weights[i]
		# calculate new means
		mean_1n = (i * mean_1 + x_i) / (i + 1)
		mean_2n = ((N - i) * mean_2 - x_i) / (N - 1 - i)

		sse_1n = sse_1 + (x_i - mean_1) ** 2 + (i + 1) * (mean_1n - mean_1) ** 2 - 2 * (mean_1n - mean_1) * (x_i - mean_1)

		sse_2n = sse_2 - (x_i - mean_2) ** 2 + (N - 1 - i) * (mean_2n - mean_2) ** 2 + 2 * (mean_2n - mean_2) * (x_i - mean_2)

		mean_1 = mean_1n
		mean_2 = mean_2n

		sse_1 = sse_1n
		sse_2 = sse_2n

		if (sse_1 + sse_2 < best_sse):
			best_sse = sse_1 + sse_2
			best_val = x_i

	return best_val

X = [0.0001, 0.0002, 0.0003, 4, 101, 102]
# print(clusterAttnWeights(np.array(X)))

