import argparse

from crf import predict
from util import read_data, read_model, score

def test(theta, data, alphabet, print_tags, print_score):
	predictions = predict(theta, data, alphabet)

	if print_tags:
		for word in predictions:
			print ''.join(word)

	if print_score:
		print score(predictions, data)

def main(test_pattern, model_dir, alphabet, print_tags, print_score):
	alphabet = list(alphabet)

	# Open feature and transition params
	theta = read_model(model_dir)

	# Read test data
	data = read_data(test_pattern)

	test(theta, data, alphabet, print_tags, print_score)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-test', help='Regex pattern of test files', dest='test', required=True)
	parser.add_argument('-model', help='Directory of model files', dest='model', required=True)
	parser.add_argument('-alphabet', help='String of all possible character labels', dest='alphabet', default='etainoshrd')
	parser.add_argument('-tag', help='Print predicted labels to stdout if 1', dest='tag', type=int, default=0)
	parser.add_argument('-score', help='Calculate and print prediction accuracy to stdout if 1', dest='score', type=int, default=1)
	args = parser.parse_args()

	main(args.test, args.model, args.alphabet, args.tag, args.score)