import crf
import util

theta = util.read_model('model')    # model directory
data = util.read_data('data/test*') # regex pattern of binarized text image files,
                                    # each sequence (word, sentence, etc.) in its own file
alphabet = list('etainoshrd')       # list of all possible character labels

predictions = crf.predict(theta, data, alphabet)
for prediction in predictions:
    print ''.join(prediction)
