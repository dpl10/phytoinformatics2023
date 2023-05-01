#!/usr/bin/env python3

### SAFE AND REQUIRED IMPORTS
import getopt
import multiprocessing
import os
import shutil
import sys
import textwrap



### CONSTANT
WRAP = shutil.get_terminal_size().columns



### PRINT TO STANDARD ERROR
def eprint(*args, **kwargs):
	print(*args, file = sys.stderr, **kwargs)

### WRAP TEXT
def eprintWrap(string, columns = WRAP):
	eprint(wrap(string, columns))

def wrap(string, columns = WRAP):
	return '\n'.join(textwrap.wrap(string, columns))



### USER SETTINGS
settings = {}
settings['batch'] = 64 
settings['cpu'] = False
settings['dataTest'] = ''
settings['encoderUnits'] = 48
settings['gpu'] = '0'
settings['model'] = ''
settings['outputArray'] = None
settings['processors'] = multiprocessing.cpu_count()
settings['sentencePieceModel'] = ''



### READ OPTIONS
arrayError = 'Number of elements in the output array (required): -a int | --array=int'
dataTestError = 'Input test data (required): -t file.tfr | --test file.tfr'
modelError = 'Input model file (required): -m file.h5 | --model=file.h5'
spmError = 'Input sentence piece model (required): -s model.pb | --spm=model.pb'
try:
	arguments, values = getopt.getopt(sys.argv[1:], 'a:b:cg:hm:p:s:t:u:', ['array=', 'batch=', 'cpu', 'gpu=', 'help', 'model=', 'processors=', 'spm', 'test=', 'units='])
except getopt.error as err:
	eprintWrap(str(err))
	sys.exit(2)
for argument, value in arguments:
	if argument in ('-a', '--array') and int(value) > 0:
		settings['outputArray'] = int(value)
	elif argument in ('-b', '--batch') and int(value) > 0:
		settings['batch'] = int(value)
	elif argument in ('-c', '--cpu'):
		settings['cpu'] = True
	elif argument in ('-g', '--gpu') and int(value) >= 0: ### does not test if device is valid
		settings['gpu'] = value
	elif argument in ('-h', '--help'):
		eprint('')
		eprintWrap('A Python3 script to test models on sequences from .tfr files with TensorFlow 2.12.0.')
		eprintWrap(arrayError)
		eprintWrap(f"Batch size (optional; default = {settings['batch']}): -b int | --batch=int")
		eprintWrap(f"CPU only (optional; default = {not settings['cpu']}): -c | --cpu")
		eprintWrap(f"Run on specified GPU (optional; default = {settings['gpu']}; CPU option overrides GPU settings): -g int | --gpu int")
		eprintWrap(modelError)
		eprintWrap(f"Processors (optional; default = {settings['processors']}): -p int | --processors=int")
		eprintWrap(spmError)
		eprintWrap(dataTestError)
		eprintWrap(f"Encoder units (optional; default = {settings['encoderUnits'] }): -u int | --units=int")
		eprint('')
		sys.exit(0)
	elif argument in ('-m', '--model'):
		if os.path.isfile(value) or os.path.isdir(value):
			settings['model'] = value
		else:
			eprintWrap(f"Model file '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-p', '--processors') and int(value) > 0:
		settings['processors'] = int(value)
	elif argument in ('-s', '--spm'):
		if os.path.isfile(value) or os.path.isdir(value):
			settings['sentencePieceModel'] = value
		else:
			eprintWrap(f"Sentence piece model file '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-t', '--test'):
		if os.path.isfile(value):
			settings['dataTest'] = value
		else:
			eprintWrap(f"Input file '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-u', '--units') and int(value) > 0:
		settings['encoderUnits'] = int(value)


### START/END
if not settings['outputArray']:
	eprintWrap(arrayError)
	sys.exit(2)
elif not settings['dataTest']:
	eprintWrap(dataTestError)
	sys.exit(2)
elif not settings['model']:
	eprintWrap(modelError)
	sys.exit(2)
elif not settings['sentencePieceModel']:
	eprintWrap(spmError)
	sys.exit(2)
else:
	eprintWrap('started...')
	for key, value in settings.items():
		eprintWrap(f"{key} = {value}")



### DISABLE OR SET GPU, THEN IMPORT TENSORFLOW
if settings['cpu'] == True:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
	os.environ['CUDA_VISIBLE_DEVICES'] = settings['gpu']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow_addons as tfa
eprintWrap(f"TensorFlow GPUs = {len(tf.config.experimental.list_physical_devices('GPU'))}")
eprintWrap(f"TensorFlow {tf.version.VERSION}\n")



### INIT
with open(settings['sentencePieceModel'], mode = 'rb') as file:
	sentencePieceModel = file.read()
sentencePieceModel = tf_text.SentencepieceTokenizer(
	model = sentencePieceModel
)



### DATASET FUNCTION
def decodeTFR(record):
	feature = {
		'sequence': tf.io.FixedLenFeature([], tf.string),
		'stability': tf.io.FixedLenFeature([], tf.float32)
	}
	record = tf.io.parse_single_example(record, feature)
	return record['sequence'], record['stability']

def tokenizer(text, labels):
	text = tf.strings.lower(text)
	tokens = sentencePieceModel.tokenize(text)
	tokens, _ = tf_text.pad_model_inputs(
		input = tokens, 
		max_seq_length = settings['encoderUnits'] , 
		pad_value = 0
	)
	return tokens, labels



### DATASETS
testData = (
	tf.data.TFRecordDataset(settings['dataTest'], compression_type = 'ZLIB')
	.map(
		decodeTFR,
		deterministic = False,
		num_parallel_calls = tf.data.AUTOTUNE
	).batch(
		batch_size = settings['batch'],
		deterministic = False,
		drop_remainder = True,
		num_parallel_calls = tf.data.AUTOTUNE
	).prefetch(tf.data.AUTOTUNE)
)
testData = testData.map(
	lambda x, y: (tokenizer(x, y)),
	deterministic = False,
	num_parallel_calls = tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)



### READ AND TEST
model = tf.keras.models.load_model(settings['model'], compile = False)
model.compile(
	loss = tf.keras.losses.MeanSquaredError(
		name = 'mse',
		reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
	),
	metrics = [
		tf.keras.metrics.LogCoshError(
			name = 'logcosh'
		),
		tf.keras.metrics.MeanSquaredError(
			name = 'mse'
		),
		tfa.metrics.RSquare(
			dtype = tf.float32,
			multioutput = 'uniform_average',
			name = 'r2',
			num_regressors = 0
		)
	],
	optimizer = tfa.optimizers.AdamW(
		learning_rate = 0.01, 
		weight_decay = 0.001
	)
)
eprint(model.summary())
stats = model.evaluate(testData)
print(f"Test loss: {stats[0]:.4f}")
print(f"Test LC: {(stats[1]):.4f}")
print(f"Test MSE: {(stats[2]):.4f}")
print(f"Test R^2: {(stats[3]):.4f}")
sys.exit(0)
