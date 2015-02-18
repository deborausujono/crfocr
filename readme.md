Usage
-----
To train a model:
```
python ocr_train.py [-h] -train TRAIN -model MODEL [-alphabet ALPHABET]
                    [-maxiter MAXITER] [-log LOG]

required arguments:
  -train TRAIN        Regex pattern of training files
  -model MODEL        Directory of model files

optional arguments:
  -h, --help          show this help message and exit
  -alphabet ALPHABET  String of all possible character labels, default = 'etainoshrd'
  -maxiter MAXITER    Maximum iteration for L-BFGS optimization, default = 1000
  -log LOG            Print log to stdout if 1, default = 1
```

For example, using the provided training data in the `data` directory:
```
python ocr_train.py -train data/train\* -model model -alphabet etainoshrd -maxiter 50 -log 0 
```

To test a model:
```
python ocr_test.py [-h] -test TEST -model MODEL [-alphabet ALPHABET]
                   [-tag TAG] [-score SCORE]

required arguments:
  -test TEST          Regex pattern of test files
  -model MODEL        Directory of model files

optional arguments:
  -h, --help          show this help message and exit
  -alphabet ALPHABET  String of all possible character labels, default = 'etainoshrd'
  -tag TAG            Print predicted labels to stdout if 1, default = 0
  -score SCORE        Calculate and print prediction accuracy to stdout if 1, default = 1
```

To run test using the provided test data and pretrained model:
```
python ocr_test.py -test data/test\* -model model
```

To load a model and use it to make predictions:
```
import crf
import string
import util

theta = util.read_model('model_directory')
data = util.read_data('filename_pattern')
alphabet = string.ascii_lowercase

predictions = crf.predict(theta, data, alphabet)
```
or see `example.py` for more details.

The dataset required by the `-train` option in `ocr_train.py`, `-test` in `ocr_test.py`, and `util.read_data` is a set of files that can be captured using the specified regular expressions. Each data case is a text file of binary pixel values of the text image and has to be in its own file. A row corresponds to a character position, where the first column of each row is the label and the remaining columns are binarized pixel values of the character image. The columns are space separated. The files can use any filename that can be captured using regular expressions. See the `data` directory for an example of naming and formatting of the training and test files. Your dataset has to follow the same format.

Running `ocr_train.py` outputs `state-params.txt` and `transition-params.txt` model files in the specified model directory. The `-model` option in `ocr_test.py` and `util.read_model` require a model directory containing `state-params.txt` and `transition-params.txt` files.

A pretrained model is included in the `model` directory, trained on 400 data cases using a limited alphabet of 10 most frequently used characters in English: “etainoshrd”. The training dataset used to obtain this model can be found in the `data` directory.

Implementation Details
----------------------
Written in Python 2.7.6, NumPy 1.9.1, and SciPy 0.15.1.

Evaluation
----------
The model obtained an accuracy of 96.87% trained on 400 and tested on 200 data cases of 10 most commonly used English letters ('etainoshrd') using the dataset provided in the `data` directory . A more detailed summary can be found in the `eval` directory.