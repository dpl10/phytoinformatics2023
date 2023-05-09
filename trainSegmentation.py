#!/usr/bin/env python3

### SAFE AND REQUIRED IMPORTS
import datetime
import getopt
import hashlib
import json
import math
import multiprocessing
import numpy as np
import os
import random
import shutil
import sys
import textwrap



### CONSTANTS
LOSSES = ('ce+clr+a', 'ce+clr+aw', 'ce+ed+aw')
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
settings['cutmix'] = False
settings['cpu'] = False
settings['dataTrain'] = ''
settings['dataValidate'] = ''
settings['epochs'] = 256
settings['gpu'] = '0'
settings['inputSize'] = 128
settings['learningRate'] = 0.005
settings['lossFunction'] = 'ce+ed+aw'
settings['model'] = ''
settings['outputArray'] = None
settings['outputDirectory'] = ''
settings['processors'] = multiprocessing.cpu_count()
settings['randomSeed'] = 123456789
settings['weightDecay'] = 0.0001



### OTHER SETTINGS
settings['analysisTime'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
settings['clrInitial'] = 4 ### try 3
settings['clrStep'] = 4 ### try 2-8
settings['leniency'] = 128
settings['randomMax'] = 2**32 ### 64 is unsafe (53 is max safe)
settings['randomMin'] = 0



### READ OPTIONS
arrayError = 'Number of elements in the output array (required): -a int | --array=int'
dataTrainError = 'Input train data (required): -t file.tfr | --train file.tfr'
dataValidateError = 'Input validation data (required): -v file.tfr | --validate file.tfr'
modelError = 'Input model file (required): -m file.h5 | --model=file.h5'
outputDirectoryError = 'Model output directory (required): -o directory | --output=directory'
try:
	arguments, values = getopt.getopt(sys.argv[1:], 'a:b:cd:e:f:g:hi:l:m:o:p:r:t:v:x', ['array=', 'batch=', 'cpu', 'decay=', 'epochs=', 'function=', 'gpu=', 'help', 'input=', 'learning=', 'model=', 'output=', 'processors=', 'random=', 'train=', 'validate=', 'cutmix'])
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
		eprintWrap('A Python3 script to train models on images from .tfr files with TensorFlow 2.12.0.')
		eprintWrap(arrayError)
		eprintWrap(f"Batch size (optional; default = {settings['batch']}): -b int | --batch=int")
		eprintWrap(f"CPU only (optional; default = {not settings['cpu']}): -c | --cpu")
		eprintWrap(f"Weight decay for AdamW optimizer (optional; default = {settings['weightDecay']}): -d float | --decay=float")
		eprintWrap(f"Number of epochs (optional; default = {settings['epochs']}): -e int | --epochs=int")
		eprintWrap(f"Loss, learning rate, and optimization function combination (optional; a = adam; aw = adamW; ce = sparse categorical cross entropy; clr = cyclical learning rate; ed = epoch decay; default = {settings['lossFunction']}): -f {'|'.join(LOSSES)} | --function={'|'.join(LOSSES)}")
		eprintWrap(f"Run on specified GPU (optional; default = {settings['gpu']}; CPU option overrides GPU settings): -g int | --gpu=int")
		eprintWrap(f"Input image size (optional; default = {settings['inputSize']}): -i int | --input=int")
		eprintWrap(f"Learning rate (optional; default = {settings['learningRate']}): -l float | --learning=float")
		eprintWrap(modelError)
		eprintWrap(outputDirectoryError)
		eprintWrap(f"Processors (optional; default = {settings['processors']}): -p int | --processors=int")
		eprintWrap(f"Random seed (optional; default = {settings['randomSeed']}): -r int | --random=int")
		eprintWrap(dataTrainError)
		eprintWrap(dataValidateError)
		# eprintWrap(f"Use cutmix augmentation (arXiv:1905.04899; optional; default = {settings['cutmix']}): -x | --cutmix")
		eprint('')
		sys.exit(0)
	elif argument in ('-i', '--input') and int(value) > 0:
		settings['inputSize'] = int(value)
	elif argument in ('-l', '--learning') and float(value) > 0.0 and float(value) < 1.0:
		settings['learningRate'] = float(value)
	elif argument in ('-m', '--model'):
		if os.path.isfile(value) or os.path.isdir(value):
			settings['model'] = value
		else:
			eprintWrap(f"Model file '{value}' does not exist!")
			sys.exit(2)
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
	elif argument in ('-t', '--train'):
		if os.path.isfile(value):
			settings['dataTrain'] = value
		else:
			eprintWrap(f"Input train file '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-v', '--validate'):
		if os.path.isfile(value):
			settings['dataValidate'] = value
		else:
			eprintWrap(f"Input validation file '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-x', '--cutmix'):
		settings['cutmix'] = True



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
import tensorflow.keras.backend as K
availableGPUs = len(tf.config.experimental.list_physical_devices('GPU'))
if settings['cpu'] == False and availableGPUs == 0:
	eprintWrap('No GPUs are available to TensorFlow. Rerun the script with -c for CPU processing only.')
	sys.exit(2)
eprintWrap(f"TensorFlow GPUs = {availableGPUs}")
eprintWrap(f"TensorFlow {tf.version.VERSION}\n")



### INIT
random.seed(settings['randomSeed'])
tf.random.set_seed(random.randint(settings['randomMin'], settings['randomMax']))

settings['halfInputSize'] = settings['inputSize']//2
settings['remainderInputSize'] = settings['inputSize']%2
ONES = tf.constant([1], dtype = tf.float32)
SHAPE = (3, 3)
ZEROS = tf.constant([0], dtype = tf.float32)



### DATASET FUNCTIONS
def batchCutMix(images, annotations, PROBABILITY = 1.0): ### based on https://www.kaggle.com/code/yihdarshieh/batch-implementation-of-more-data-augmentations/notebook?scriptVersionId=29767726 and https://www.kaggle.com/code/cdeotte/cutmix-and-mixup-on-gpu-tpu
	if settings['cutmix']:
		new = tf.cast(tf.random.uniform([settings['batch']], 0, settings['batch']), tf.int32)
		x = tf.cast(tf.random.uniform([settings['batch']], 0, settings['inputSize']), tf.int32)
		y = tf.cast(tf.random.uniform([settings['batch']], 0, settings['inputSize']), tf.int32)
		b = tf.random.uniform([settings['batch']], 0, 1) ### beta distribution with alpha = 1.0
		width = tf.cast(settings['inputSize'] * tf.math.sqrt(1-b), tf.int32) * tf.cast(tf.random.uniform([settings['batch']], 0, 1) <= PROBABILITY, tf.int32)
		width //= 2
		xa = tf.math.maximum(0, x-width)
		xb = tf.math.minimum(settings['inputSize'], x+width)
		ya = tf.math.maximum(0, y-width)
		yb = tf.math.minimum(settings['inputSize'], y+width)
		mixupImages = cutMix(
			images = images, 
			new = new, 
			xa = xa, 
			xb = xb, 
			ya = ya, 
			yb = yb
		)
		mixupAnnotations = cutMix(
			images = annotations, 
			new = new, 
			xa = xa, 
			xb = xb, 
			ya = ya, 
			yb = yb
		)
		return mixupImages, mixupAnnotations
	else:
		return images, annotations

def batchMix(images, annotations, PROBABILITY = 1.0): ### based on https://www.kaggle.com/code/yihdarshieh/batch-implementation-of-more-data-augmentations/notebook?scriptVersionId=29767726 and https://www.kaggle.com/code/cdeotte/cutmix-and-mixup-on-gpu-tpu
	mixup = tf.cast(tf.random.uniform([settings['batch']], 0, 1) <= PROBABILITY, tf.int32)
	mix = tf.random.uniform([settings['batch']], 0, 1)*tf.cast(mixup, tf.float32) ### beta distribution with alpha = 1.0
	new = tf.cast(tf.random.uniform([settings['batch']], 0, settings['batch']), tf.int32)
	mixupImages = (1-mix)[:, tf.newaxis, tf.newaxis, tf.newaxis] * images + mix[:, tf.newaxis, tf.newaxis, tf.newaxis] * tf.gather(images, new)
	mixupLabels = (1-mix)[:, tf.newaxis, tf.newaxis, tf.newaxis] * annotations + mix[:, tf.newaxis, tf.newaxis, tf.newaxis] * tf.gather(annotations, new)
	return mixupImages, mixupLabels

def cutMix(images, new, xa, xb, ya, yb):
	one = images[:, ya:yb, 0:xa, :]
	two = images[new, ya:yb, xa:xb, :]
	three = images[:, ya:yb, xb:settings['inputSize'], :]
	middle = tf.concat([one, two, three], axis = 1)
	return tf.concat([images[:, 0:ya, :, :], middle, images[:, yb:settings['inputSize'], :, :]], axis = 0)

def decodeTFR(record):
	feature = {
		'annotation': tf.io.FixedLenFeature([], tf.string),
		'image': tf.io.FixedLenFeature([], tf.string)
	}
	record = tf.io.parse_single_example(record, feature)
	record['annotation'] = tf.cast(tf.io.decode_png(
		channels = 1,
		contents = record['annotation']
	), tf.float32)
	record['annotation'].set_shape((settings['inputSize'], settings['inputSize'], 1)) ### forces tf to recall the size later
	record['image'] = tf.cast(tf.io.decode_jpeg(
		channels = 3,
		contents = record['image']
	), tf.float32)
	record['image'].set_shape((settings['inputSize'], settings['inputSize'], 3)) ### forces tf to recall the size later
	return record['image'], record['annotation']

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
#
# fix code with https://github.com/tensorflow/transform/issues/169
# 		
	# ).map(
	# 	lambda x, y: (batchCutMix(x, y)),
	# 	deterministic = False,
	# 	num_parallel_calls = tf.data.AUTOTUNE
#
# code works, but not a good idea for learning
#
	# ).map(
	# 	lambda x, y: (batchMix(x, y)),
	# 	deterministic = False,
	# 	num_parallel_calls = tf.data.AUTOTUNE
	).prefetch(tf.data.AUTOTUNE)
)



### READ AND TRAIN MODEL
def epoch2decayLR(epoch, learningRate):
	return learningRate*1.0/(1.0+(settings['learningRate']/settings['epochs'])*epoch)

model = tf.keras.models.load_model(settings['model'], compile = False)
eprint(model.summary())

### loss
loss = tf.keras.losses.SparseCategoricalCrossentropy(
	from_logits = True
)

### metrics
metrics = [
	tf.keras.metrics.SparseCategoricalAccuracy(
		name = 'accuracy'
	),
	tf.keras.metrics.MeanIoU(
		axis = -1,
		dtype = None,
		ignore_class = 0,
		name = 'MeanIoU',
		num_classes = settings['outputArray'],
		sparse_y_pred = False,
		sparse_y_true = True
	)
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
if settings['lossFunction'] == 'ce+clr+a':
	optimizer = tf.keras.optimizers.Adam(
		learning_rate = clr
	)
elif settings['lossFunction'] == 'ce+clr+aw':
	optimizer = tfa.optimizers.AdamW(
		learning_rate = clr,
		weight_decay = settings['weightDecay']
	)
elif settings['lossFunction'] == 'ce+ed+aw':
	optimizer = tfa.optimizers.AdamW(
		learning_rate = settings['learningRate'], 
		weight_decay = settings['weightDecay']
	)

### compile
model.compile(
	loss = loss,
	metrics = metrics,
	optimizer = optimizer
)

### callbacks
callbacks = []
callbacks.append(tf.keras.callbacks.EarlyStopping(
	baseline = None,
	min_delta = 0,
	mode = 'max',
	monitor = 'MeanIoU',
	patience = settings['leniency'],
	restore_best_weights = True,
	verbose = 0
))
if settings['lossFunction'] == 'ce+ed+aw':
	callbacks.append(tf.keras.callbacks.LearningRateScheduler(epoch2decayLR))

### train
history = model.fit(
	batch_size = settings['batch'],
	callbacks = callbacks,
	epochs = settings['epochs'],
	validation_data = validationData,
	x = trainData
)

### save
encoded = json.dumps(settings, ensure_ascii = False, indent = 3, sort_keys = True).encode()
hexMD5 = hashlib.md5(encoded).hexdigest()
model.save(os.path.join(settings['outputDirectory'], f"best-{hexMD5}", 'best-model.h5')) 
np.save(os.path.join(settings['outputDirectory'], f"best-{hexMD5}", 'training-history.npy'), history.history, allow_pickle = True) ### history = np.load('file', allow_pickle = True).item()
with open(os.path.join(settings['outputDirectory'], f"best-{hexMD5}", 'training-settings.json'), 'w') as file:
	print(encoded, file = file)

sys.exit(0)
