#!/usr/bin/env python3

### SAFE AND REQUIRED IMPORTS
import datetime
import getopt
import hashlib
import json
import multiprocessing
import numpy as np
import os
import random
import shutil
import sys
import textwrap



### CONSTANTS
LOSSES = ('lc+clr+a', 'lc+clr+aw', 'lc+ed+aw', 'lc+clr+sgd', 'mse+clr+a', 'mse+clr+aw', 'mse+ed+aw', 'mse+clr+sgd')
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
settings['dataTrain'] = ''
settings['dataValidate'] = ''
settings['encoderUnits'] = 48
settings['epochs'] = 256
settings['gpu'] = '0'
settings['learningRate'] = 0.005
settings['lossFunction'] = 'mse+clr+aw'
settings['model'] = ''
settings['nta'] = False
settings['outputArray'] = None
settings['outputDirectory'] = ''
settings['processors'] = multiprocessing.cpu_count()
settings['randomSeed'] = 123456789
settings['sentencePieceModel'] = ''
settings['weightDecay'] = 0.0001



### OTHER SETTINGS
settings['analysisTime'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
settings['clrInitial'] = 4 ### try 3
settings['clrStep'] = 4 ### try 2-8
settings['initializer'] = 'glorot_uniform'
settings['leniency'] = 128
settings['randomMax'] = 2**32 ### 64 is unsafe (53 is max safe)
settings['randomMin'] = 0



### READ OPTIONS
arrayError = 'Number of elements in the output array (required): -a int | --array=int'
dataTrainError = 'Input train data (required): -t file.tfr | --train file.tfr'
dataValidateError = 'Input validation data (required): -v file.tfr | --validate file.tfr'
modelError = 'Input model file (required): -m file.h5 | --model=file.h5'
outputDirectoryError = 'Model output directory (required): -o directory | --output=directory'
spmError = 'Input sentence piece model (required): -s model.pb | --spm=model.pb'
try:
	arguments, values = getopt.getopt(sys.argv[1:], 'a:b:cd:e:f:g:hl:m:no:p:r:s:t:u:v:', ['array=', 'batch=', 'cpu', 'decay=', 'epochs=', 'function=', 'gpu=', 'help', 'learning=', 'model=', 'nta', 'output=', 'processors=', 'random=', 'spm=', 'train=', 'units=', 'validate='])
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
	elif argument in ('-d', '--decay') and float(value) > 0.0 and float(value) < 1.0:
		settings['weightDecay'] = float(value)
	elif argument in ('-e', '--epochs') and int(value) > 0:
		settings['epochs'] = int(value)
	elif argument in ('-f', '--function') and value in LOSSES:
		settings['lossFunction'] = value	
	elif argument in ('-g', '--gpu') and int(value) >= 0: ### does not test if device is valid
		settings['gpu'] = value
	elif argument in ('-h', '--help'):
		eprint('')
		eprintWrap('A Python3 script to train models on images from .tfr files with TensorFlow 2.12.0')
		eprintWrap(arrayError)
		eprintWrap(f"Batch size (optional; default = {settings['batch']}): -b int | --batch=int")
		eprintWrap(f"CPU only (optional; default = {not settings['cpu']}): -c | --cpu")
		eprintWrap(f"Weight decay for AdamW optimizer (optional; default = {settings['weightDecay']}): -d float | --decay=float")
		eprintWrap(f"Number of epochs (optional; default = {settings['epochs']}): -e int | --epochs=int")
		eprintWrap(f"Loss, learning rate, and optimization function combination (optional; a = adam; aw = adamW; clr = cyclical learning rate; ed = epoch decay; lc = logarithm of the hyperbolic cosine error; mse = mean squared error; sgd = stochastic gradient descent; default = {settings['lossFunction']}): -f {'|'.join(LOSSES)} | --function={'|'.join(LOSSES)}")
		eprintWrap(f"Run on specified GPU (optional; default = {settings['gpu']}; CPU option overrides GPU settings): -g int | --gpu=int")
		eprintWrap(f"Learning rate (optional; default = {settings['learningRate']}): -l float | --learning=float")
		eprintWrap(modelError)
#
# -n
# 		
		eprintWrap(outputDirectoryError)
		eprintWrap(f"Processors (optional; default = {settings['processors']}): -p int | --processors=int")
		eprintWrap(f"Random seed (optional; default = {settings['randomSeed']}): -r int | --random=int")
		eprintWrap(spmError)
		eprintWrap(dataTrainError)
		eprintWrap(f"Encoder units (optional; default = {settings['encoderUnits'] }): -u int | --units=int")
		eprintWrap(dataValidateError)
		eprint('')
		sys.exit(0)
	elif argument in ('-l', '--learning') and float(value) > 0.0 and float(value) < 1.0:
		settings['learningRate'] = float(value)
	elif argument in ('-m', '--model'):
		if os.path.isfile(value) or os.path.isdir(value):
			settings['model'] = value
		else:
			eprintWrap(f"Model file '{value}' does not exist!")
			sys.exit(2)
#
# -n
# 			
	elif argument in ('-o', '--output'):
		if os.path.isdir(value):
			settings['outputDirectory'] = value
		else:
			eprintWrap(f"Model output directory '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-p', '--processors') and int(value) > 0:
		settings['processors'] = int(value)
	elif argument in ('-r', '--random') and int(value) >= settings['randomMin'] and int(value) <= settings['randomMax']:
		settings['randomSeed'] = int(value)
	elif argument in ('-s', '--spm'):
		if os.path.isfile(value) or os.path.isdir(value):
			settings['sentencePieceModel'] = value
		else:
			eprintWrap(f"Sentence piece model file '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-t', '--train'):
		if os.path.isfile(value):
			settings['dataTrain'] = value
		else:
			eprintWrap(f"Input train file '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-u', '--units') and int(value) > 0:
		settings['encoderUnits'] = int(value)
	elif argument in ('-v', '--validate'):
		if os.path.isfile(value):
			settings['dataValidate'] = value
		else:
			eprintWrap(f"Input validation file '{value}' does not exist!")
			sys.exit(2)



### START/END
if not settings['outputArray']:
	eprintWrap(arrayError)
	sys.exit(2)
elif not settings['dataValidate']:
	eprintWrap(dataValidateError)
	sys.exit(2)
elif not settings['dataTrain']:
	eprintWrap(dataTrainError)
	sys.exit(2)
elif not settings['model']:
	eprintWrap(modelError)
	sys.exit(2)
elif not settings['outputDirectory']:
	eprintWrap(outputDirectoryError)
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
import tensorflow_addons as tfa
import tensorflow_text as tf_text
availableGPUs = len(tf.config.experimental.list_physical_devices('GPU'))
if settings['cpu'] == False and availableGPUs == 0:
	eprintWrap('No GPUs are available to TensorFlow. Rerun the script with -c for CPU processing only.')
	sys.exit(2)
eprintWrap(f"TensorFlow GPUs = {availableGPUs}")
eprintWrap(f"TensorFlow {tf.version.VERSION}\n")



### INIT
random.seed(settings['randomSeed'])
tf.random.set_seed(random.randint(settings['randomMin'], settings['randomMax']))

with open(settings['sentencePieceModel'], mode = 'rb') as file:
	sentencePieceModel = file.read()
sentencePieceModel = tf_text.SentencepieceTokenizer(
	model = sentencePieceModel
)



### DATASET FUNCTIONS
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
validationData = (
	tf.data.TFRecordDataset(settings['dataValidate'], compression_type = 'ZLIB')
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
validationData = validationData.map(
	lambda x, y: (tokenizer(x, y)),
	deterministic = False,
	num_parallel_calls = tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

trainData = (
	tf.data.TFRecordDataset(settings['dataTrain'], compression_type = 'ZLIB')
	.map(
		decodeTFR,
		deterministic = False,
		num_parallel_calls = tf.data.AUTOTUNE
	).shuffle(
		buffer_size = settings['batch']*100,
		reshuffle_each_iteration = True
	).batch(
		batch_size = settings['batch'],
		deterministic = False,
		drop_remainder = True,
		num_parallel_calls = tf.data.AUTOTUNE
	).prefetch(tf.data.AUTOTUNE)
)
trainData = trainData.map(
	lambda x, y: (tokenizer(x, y)),
	deterministic = False,
	num_parallel_calls = tf.data.AUTOTUNE
# ).map(
# 	lambda x, y: (batchNTA(x, y)),
# 	deterministic = False,
# 	num_parallel_calls = tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)



### READ AND TRAIN MODEL
def epoch2decayLR(epoch, learningRate):
	return learningRate*1.0/(1.0+(settings['learningRate']/settings['epochs'])*epoch)

model = tf.keras.models.load_model(settings['model'], compile = False)
modelTruncated = tf.keras.Model(inputs = model.input, outputs = model.layers[-2].output)
output = tf.keras.layers.Dense(
	activation = 'tanh',
	activity_regularizer = None,
	bias_constraint = None,
	bias_initializer = 'zeros',
	bias_regularizer = None,
	kernel_constraint = None,
	kernel_initializer = settings['initializer'],
	kernel_regularizer = None,
	name = 'output_dense',
	units = settings['outputArray'],
	use_bias = True
)(modelTruncated.layers[-1].output)
newModel = tf.keras.Model(inputs = modelTruncated.input, outputs = output)
for layer in newModel.layers:
	layer.trainable = False
newModel.layers[-2].trainable = True
newModel.layers[-1].trainable = True
eprint(newModel.summary())

#
#
#
encoded = json.dumps(settings, ensure_ascii = False, indent = 3, sort_keys = True).encode()
hexMD5 = hashlib.md5(encoded).hexdigest()
newModel.save(os.path.join(settings['outputDirectory'], f"best-{hexMD5}", 'new-model.h5')) 
#
# 
#


### loss
loss = None
if settings['lossFunction'] in ('lc+clr+a', 'lc+clr+aw', 'lc+ed+aw', 'lc+clr+sgd'): 
	loss = tf.keras.losses.LogCosh(
		reduction = tf.keras.losses.Reduction.SUM
	)
elif settings['lossFunction'] in  ('mse+clr+a', 'mse+clr+aw', 'mse+ed+aw', 'mse+clr+sgd'):
	loss = tf.keras.losses.MeanSquaredError(
		name = 'mse',
		reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
	)

### metrics
metrics = [
	tf.keras.metrics.LogCoshError(
		name = 'logcosh'
	),
	tf.keras.metrics.MeanSquaredError(
		name = 'mse'
	),
	# tfa.metrics.RSquare(
	# 	multioutput = 'uniform_average',
	# 	name = 'r2',
	# 	num_regressors = 0
	# )
]

### optimizer
clr = tfa.optimizers.CyclicalLearningRate(
	initial_learning_rate = settings['learningRate']/settings['clrInitial'],
	maximal_learning_rate = settings['learningRate'],
	name = 'CyclicalLearningRate',
	scale_fn = lambda x: 1/(2.0**(x-1)),
	scale_mode = 'cycle',
	step_size = settings['clrStep']*settings['batch'] 
)
optimizer = None
if settings['lossFunction'] in ('lc+clr+a', 'mse+clr+a'):
	optimizer = tf.keras.optimizers.Adam(
		learning_rate = clr
	)
elif settings['lossFunction'] in ('lc+clr+aw', 'mse+clr+aw'):
	optimizer = tfa.optimizers.AdamW(
		learning_rate = clr,
		weight_decay = settings['weightDecay']
	)
elif settings['lossFunction'] in ('lc+ed+aw', 'mse+ed+aw'):
	optimizer = tfa.optimizers.AdamW(
		learning_rate = settings['learningRate'], 
		weight_decay = settings['weightDecay']
	)
elif settings['lossFunction'] in ('lc+clr+sgd', 'mse+clr+sgd'):
	optimizer = tf.keras.optimizers.SGD(
		learning_rate = clr
	)

### compile
newModel.compile(
	loss = loss,
	metrics = metrics,
	optimizer = optimizer
)

### callbacks
callbacks = []
callbacks.append(tf.keras.callbacks.EarlyStopping(
	baseline = None,
	min_delta = 0,
	mode = 'min',
	monitor = f"val_{'logcosh' if settings['lossFunction'] in ('lc+clr+a', 'lc+clr+aw', 'lc+ed+aw', 'lc+clr+sgd') else 'mse'}",
	patience = settings['leniency'],
	restore_best_weights = True,
	verbose = 0
))
if settings['lossFunction'] in ('lc+ed+aw', 'mse+ed+aw'):
	callbacks.append(tf.keras.callbacks.LearningRateScheduler(epoch2decayLR))

# for sample in validationData.take(2):
# 	print(sample[0])
# 	print(sample[1])

### train
history = newModel.fit(
	batch_size = settings['batch'],
	callbacks = callbacks,
	epochs = settings['epochs'],
	validation_data = validationData,
	x = trainData
)

### save
encoded = json.dumps(settings, ensure_ascii = False, indent = 3, sort_keys = True).encode()
hexMD5 = hashlib.md5(encoded).hexdigest()
newModel.save(os.path.join(settings['outputDirectory'], f"best-{hexMD5}", 'best-model.h5')) 
np.save(os.path.join(settings['outputDirectory'], f"best-{hexMD5}", 'training-history.npy'), history.history, allow_pickle = True) ### history = np.load('file', allow_pickle = True).item()
with open(os.path.join(settings['outputDirectory'], f"best-{hexMD5}", 'training-settings.json'), 'w') as file:
	print(encoded, file = file)

sys.exit(0)
