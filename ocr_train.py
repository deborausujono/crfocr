import argparse
import os
import time

import numpy as np

from crf import likelihood, likelihood_prime
from scipy.optimize import fmin_l_bfgs_b
from util import read_data, print_model

def train(data, alphabet, maxiter, log):
	"""
	Returns the learned [state_params, trans_params] list,
	where each parameter table is a numpy array.
	"""

	# Initialize state and transition parameter tables with zeros
	state_params = np.ndarray.flatten(np.zeros((len(alphabet), len(data[0][1][0]))))
	trans_params = np.ndarray.flatten(np.zeros((len(alphabet), len(alphabet))))
	theta = np.concatenate([state_params, trans_params])

	# Learn by minimizing the negative average log likelihood
	t0 = time.time()
	theta, fmin, _ = fmin_l_bfgs_b(likelihood, theta, fprime=likelihood_prime,
		                           args=(data, alphabet), maxiter=maxiter, disp=log)
	t1 = time.time()

	# Write training summary to log
	if log > 0:
		print "Training data size:", len(data)
		print "Value of likelihood function at minimum:", np.exp(-fmin)
		print "Training time:", t1-t0

	k = len(alphabet)
	n = len(data[0][1][0])
	mid = k * n
	state_params = np.reshape(theta[:mid], (k, n))
	trans_params = np.reshape(theta[mid:], (k, k))

	return [state_params, trans_params]

def main(train_pattern, model_dir, alphabet, maxiter, log):
	alphabet = list(alphabet)

	# Read training data
	data = read_data(train_pattern)
	if log > 0:
		print 'Successfully read', len(data), 'data cases'

	# Train the model
	model = train(data, alphabet, maxiter, log)

	# Save the model
	if log > 0:
		print 'Saving model to', model_dir
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)
	state_file = model_dir + "/state-params.txt"
	trans_file = model_dir + "/transition-params.txt"
	print_model(model, state_file, trans_file)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-train', help='Regex pattern of training files', dest='train', required=True)
	parser.add_argument('-model', help='Directory of model files', dest='model', required=True)
	parser.add_argument('-alphabet', help='String of all possible character labels', dest='alphabet', default='etainoshrd')
	parser.add_argument('-maxiter', help='Maximum iteration for L-BFGS optimization', dest='maxiter', default=1000, type=int)
	parser.add_argument('-log', help='Print log to stdout if 1', dest='log', default=1, type=int)
	args = parser.parse_args()

	main(args.train, args.model, args.alphabet, args.maxiter, args.log)
