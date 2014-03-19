import sys
import time
import numpy as np

from scipy.misc import logsumexp
from scipy.optimize import fmin_l_bfgs_b

def get_true_labels(fn):
	f = open(fn, 'r')
	true_labels = [list(x) for x in f.read().splitlines()]
	return true_labels

def get_data(fprefix, n):
	data = []
	for i in range(1, n+1):
		fn = fprefix+str(i)+".txt"
		f = open(fn, 'r')
		lines = f.readlines()
		word = []
		for line in lines:
			word.append([int(x) for x in line.split()])
		data.append(word)
	
	return data

"""Returns an nx10 numpy array, where n is the word length.
Each row corresponds to a position in the word, each column to a label (e, t, a, ...), and
each cell is the node potential. Takes two arguments: word, which is an nx321 matrix,
where each row represents a position in the word and each cell is a feature value; and
feature_params, which is a 10x321 matrix, where each row corresponds to a character label
(e, t, a, ...), each column to a feature, and the cell is the feature parameter value."""
def node_potentials(word, feature_params):
	phi = []
	for char in word:
		v = []
		for char_param in feature_params:
			potential = np.sum(np.multiply(char, char_param))
			v.append(potential)
		phi.append(v)

	#print phi #Uncomment to print
	return np.array(phi)

"""Computes the clique potentials of a single clique in a chain.
Returns a 10x10 numpy array, where each cell is the clique potential of Yi, Yi+1.
Each row corresponds to Yi for each possible label (e, t, ...),
and each column corresponds to Yi+1 also for each possible label."""
def single_clique_potentials(node_factor1, node_factor2, trans_params):
	clique_potentials = trans_params + node_factor1[:, np.newaxis]
	if node_factor2 is not None:
		clique_potentials = clique_potentials + node_factor2

	#print clique_potentials #Uncomment to print
	return clique_potentials

"""Computes the clique potentials of the entire chain.
Returns an (n-1)x10x10 numpy array, where n is the word length.
Takes two arguments: x is a list of [feature_params, trans_params], and word is an nx321 matrix."""
def chain_clique_potentials(x, word):
	feature_params = x[0]
	trans_params = x[1]

	phi = node_potentials(word, feature_params)

	psi = []
	i = 0
	while i < len(word)-1:
		node_factor1 = phi[i]
		node_factor2 = None
		if i == len(word)-2:
			node_factor2 = phi[i+1]
		psi.append(single_clique_potentials(node_factor1, node_factor2, trans_params))
		i += 1

	#Uncomment below to print
	"""for clique in psi:
		print clique"""

	return np.array(psi)

"""Returns an (n-2)x10 numpy array, where n is the word length.
Each row is the message from clique i to clique i+1, and each column corresponds to each possible label.
Takes psi as an argument, which is a numpy array of 10x10 clique potentials."""
def sum_product_messages(psi):
	delta = []
	clique_len = len(psi) #Number of cliques in the word
	i = clique_len-1
	
	#Backward messages
	prev_msgs = np.zeros(len(psi[0])) #[0] * len(psi[0])
	while i > 0:
		msg = logsumexp(psi[i] + prev_msgs, axis=1)
		delta.append(msg)
		#print msg #Uncomment to print
		prev_msgs = prev_msgs + msg
		i -= 1

	#Forward messages
	prev_msgs = np.zeros(len(psi[0])) #[0] * len(psi[0])
	while i < clique_len-1:
		msg = logsumexp(psi[i] + prev_msgs[:, np.newaxis], axis=0)
		delta.append(msg)
		#print msg #Uncomment to print
		prev_msgs = prev_msgs + msg
		i += 1

	return np.array(delta)

"""Returns a numpy array of size (n-1)x10x10, where n is the word length.
The array contains n-1 log belief tables of size 10x10 for each clique in the word.
Takes two arguments: x, which is a list of [feature_params, trans_params]; and word, which is
an nx321 matrix representing the word image."""
def beliefs(x, word):
	psi = chain_clique_potentials(x, word)
	delta = sum_product_messages(psi)

	beta = []
	mid = len(delta)/2-1
	bwd_idx = mid
	fwd_idx = mid

	i = 0
	while i < len(psi):
		belief = None
		if i == 0: #bwd only
			belief = psi[i] + delta[bwd_idx]
		elif i == len(psi)-1: #fwd only
			belief = psi[i] + delta[fwd_idx][:, np.newaxis] 
		else:
			belief = psi[i] + delta[fwd_idx][:, np.newaxis] + delta[bwd_idx]
		
		beta.append(belief)
		fwd_idx += 1
		bwd_idx -= 1

		#Uncomment below to print e, t
		"""print_order = [0, 1]
		for j in print_order:
			for k in print_order:
				print belief[j][k],
			print"""

		i += 1

	return np.array(beta)

"""Computes the pairwise marginal probabilities. Returns a numpy array of size (n-1)x10x10,
where n is the word length. Each 10x10 table is the pairwise probability table for each of
the (n-1) cliques (i.e., transition probability from Yi to Yi+1).
Takes beta, a numpy array of log belief tables for each clique, as an argument."""
def get_pairwise_p(beta):
	p_table = []
	for clique_belief in beta:
		denom = logsumexp(clique_belief)
		p = np.exp(clique_belief-denom)
		p_table.append(p)

		#Uncomment to print e, t, r
		"""print_order = [0, 1, 8]
		for j in print_order:
			for k in print_order:
				print p[j][k],
			print
		"""

	return np.array(p_table)

"""Computes the single-variable marginal probabilities.
Returns an nx10 numpy array, where n is the word length.
Each row corresponds to a position in the word, and each column represents a label (e, t, a, ...)."""
def get_single_p(pairwise_p):
	p_table = []
	i = 0
	for trans in pairwise_p:
		p_table.append(np.sum(trans, axis=1))
		if i == len(pairwise_p)-1:
			p_table.append(np.sum(trans, axis=0))
		i += 1

	#Uncomment below to print values
	"""for pos in p_table:
		print pos"""

	return np.array(p_table)

"""Returns the joint probability of a sequence of labels."""
def joint_p(single_p, labels, char_order):
	p_per_pos = []
	for (label, p) in zip(labels, single_p):
		idx = char_order.index(label)
		p_per_pos.append(np.log(p[idx]))

	return np.sum(p_per_pos)

"""Objective function to be minimized. Returns the negated average log likelihood value
given the true labels from training data."""
def likelihood(x, data, true_labels, char_order):
	#If flattened, reshape vector x into a feature params table of size 321x10,
	#and a transition params table of size 10x10
	if len(x) != 2:
		label_len = len(char_order)
		feature_len = len(data[0][0])
		mid = label_len*feature_len

		feature_params = np.reshape(x[:mid], (label_len, feature_len))
		trans_params = np.reshape(x[mid:], (label_len, label_len))
		x = [feature_params, trans_params]

	p_per_word = []
	for (word, labels) in zip(data, true_labels):
		beta = beliefs(x, word)
		pairwise_p = get_pairwise_p(beta)
		single_p = get_single_p(pairwise_p)
		p_per_word.append(joint_p(single_p, labels, char_order))

	#print np.exp(np.sum(p_per_word)/len(data)) #Uncomment to print
	return -np.sum(p_per_word)/len(data)

"""Returns a flattened vector of a 10x321 numpy array with every element negated"""
def feature_gradient(x, data, true_labels, char_order):
	#Initialize a 10x321 feature gradient table of zeros
	gradient = np.zeros((len(char_order), len(data[0][0])))

	for (word, labels) in zip(data, true_labels):
		beta = beliefs(x, word)
		pairwise_p = get_pairwise_p(beta)
		single_p = get_single_p(pairwise_p)
		for (pos, label, p_per_pos) in zip(word, labels, single_p):
			for i in range(len(gradient)): #possible labels
				for j in range(len(gradient[i])): #features
					indicator = 0
					if label == char_order[i]:
						indicator = 1
					gradient[i][j] += (indicator - p_per_pos[i])*pos[j]
	
	gradient = gradient/len(data)

	return np.ndarray.flatten(np.negative(gradient))

"""Returns a flattened vector of a 10x10 numpy array with every element negated"""
def transition_gradient(x, data, true_labels, char_order):
	#Initialize a 10x10 transition gradient table of zeros
	gradient = np.zeros((len(char_order), len(char_order)))

	for (word, labels) in zip(data, true_labels):
		beta = beliefs(x, word)
		pairwise_p = get_pairwise_p(beta)
		label_pairs = zip([None] + labels, labels + [None])[1:-1]

		for (label_pair, p_per_pair) in zip(label_pairs, pairwise_p):
			(label1, label2) = label_pair

			for i in range(len(gradient)):
				for j in range(len(gradient)):
					indicator = 0
					if label1 == char_order[i] and label2 == char_order[j]:
						indicator = 1
					gradient[i][j] += indicator - p_per_pair[i][j]

	gradient = gradient/len(data)

	return np.ndarray.flatten(np.negative(gradient))

"""Returns a flattened vector of the list [feature_gradient, transition_gradient]"""
def likelihood_prime(x, data, true_labels, char_order):
	#Reshape flattened vectors into a 10x321 feature param matrix and a 10x10 trans param matrix
	label_len = len(char_order)
	feature_len = len(data[0][0])
	mid = label_len*feature_len

	feature_params = np.reshape(x[:mid], (label_len, feature_len))
	trans_params = np.reshape(x[mid:], (label_len, label_len))
	x = [feature_params, trans_params]

	return np.concatenate([feature_gradient(x, data, true_labels, char_order), transition_gradient(x, data, true_labels, char_order)])

"""Prints out feature and transition parameters to file. Takes three arguments:
x is a flat vector of length (10*321+10*10), which is the output of the optimizer;
feature_params_fn and trans_params_fn is the filename for feature and transition params output
respectively."""
def print_params(x, feature_fn, trans_fn):
	fns = [feature_fn, trans_fn]
	for (params, fn) in zip(x, fns):
		f = open(fn, 'w')
		for row in params:
			for cell in row:
				f.write(str(cell)+" ")
			f.write("\n")
		f.close()

"""Returns a list of [feature_params, trans_params]. Each param table is a numpy array."""
def train(data, true_labels, char_order):
	#Create a log file
	f = open("train_out.txt", 'a')

	#Initialize feature and transition parameter tables of zeros for initial guess
	feature_params = np.ndarray.flatten(np.zeros((len(char_order), len(data[0][0]))))
	trans_params = np.ndarray.flatten(np.zeros((len(char_order), len(char_order))))
	x = np.concatenate([feature_params, trans_params])

	#Minimize negative average log likelihood
	t0 = time.time()
	optimize_out = fmin_l_bfgs_b(likelihood, x, fprime=likelihood_prime, args=(data, true_labels, char_order), maxiter=30)
	t1 = time.time()

	#Write info to log
	f.write("Data size = "+str(len(data))+"\n")
	f.write("Likelihood function value at minimum = "+str(np.exp(-optimize_out[1]))+"\n")
	f.write("Total time = " + str(t1-t0)+"\n")
	f.write("=====================================================\n")
	f.close()

	x = optimize_out[0]
	label_len = len(char_order)
	feature_len = len(data[0][0])
	mid = label_len*feature_len
	feature_params = np.reshape(x[:mid], (label_len, feature_len))
	trans_params = np.reshape(x[mid:], (label_len, label_len))

	return [feature_params, trans_params]

def main():
	#Index of chars in the order as specified in the assignment writeup
	char_order = list("etainoshrd")

	#Get training data
	fprefix = sys.argv[1] #"data/train_img"
	n = int(sys.argv[2]) #50
	data = get_data(fprefix, n)

	#Get true labels
	true_labels = get_true_labels(sys.argv[3]) #"data/train_words.txt"

	x = train(data, true_labels, char_order)

	#Print out learned params to file
	feature_fn = "feature-params-"+str(n)+".txt"
	trans_fn = "transition-params-"+str(n)+".txt"
	print_params(x, feature_fn, trans_fn)

if __name__ == '__main__':
	main()
