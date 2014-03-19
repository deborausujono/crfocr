import sys
import numpy as np

from crf_train import beliefs, likelihood
from crf_train import get_pairwise_p, get_single_p
from crf_train import get_data, get_true_labels

def get_params(feature_fprefix, trans_fprefix, n0, n1, nstep):
	x = []
	fprefixes = [feature_fprefix, trans_fprefix]

	for n in range(n0, n1+1, nstep):
		x0 = []
		for fprefix in fprefixes:
			f = open(fprefix+str(n)+".txt", 'r')
			lines = f.readlines()
			param_table = []
			for line in lines:
				param_table.append([float(y) for y in line.split()])
			x0.append(param_table)
		x.append(x0)

	return x

def predict_word(single_p, char_order):
	word = []
	for pos in single_p:
		l = pos
		word.append(char_order[np.argmax(pos)])
	return word

def get_predictions(x, data, char_order):
	predictions = []

	for x0 in x:
		predictions0 = []
		for word in data:
			beta = beliefs(x0, word)
			pairwise_p = get_pairwise_p(beta)
			single_p = get_single_p(pairwise_p)
			predictions0.append(predict_word(single_p, char_order))
		#print predictions0[:10] #Uncomment to print
		predictions.append(predictions0)

	return predictions

def test(x, data, true_labels, char_order):
	predictions = get_predictions(x, data, char_order)

	#Compute error
	for predictions0 in predictions:
		correct_count = 0
		char_count = 0
		for (prediction, label) in zip(predictions0, true_labels):
			for (char, true_char) in zip(prediction, label):
				if char == true_char:
					correct_count += 1
				char_count += 1

		accuracy = 1.0*correct_count/char_count
		print accuracy, 1-accuracy

	#Compute average conditional log likelihood
	for x0 in x:
		l = -likelihood(x0, data, true_labels, char_order)
		print l, np.exp(l)

def main():
	#Index of chars in the order as specified in the assignment writeup
	char_order = list("etainoshrd")

	#Open feature and transition params
	feature_fprefix = sys.argv[1] #"model/feature-params-"
	trans_fprefix = sys.argv[2] #"model/transition-params-"
	model_n0 = int(sys.argv[3])
	model_n1 = int(sys.argv[4])
	model_nstep = int(sys.argv[5])
	x = get_params(feature_fprefix, trans_fprefix, model_n0, model_n1, model_nstep)

	#Get data
	fprefix = sys.argv[6] #"data/test_img"
	n =  int(sys.argv[7]) #200
	data = get_data(fprefix, n)

	#Get true labels
	true_labels = get_true_labels(sys.argv[8]) #"data/test_words.txt"

	test(x, data, true_labels, char_order)

if __name__ == '__main__':
	main()