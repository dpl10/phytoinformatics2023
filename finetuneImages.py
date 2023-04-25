#!/usr/bin/env python3

### SAFE AND REQUIRED IMPORTS
import ast
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
LOSSES = ('ce+clr+a', 'ce+clr+aw', 'ce+ed+aw', 'ce+clr+sgd')
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
settings['classCount'] = None
settings['classWeight'] = None
settings['cpu'] = False
settings['cutmix'] = False
settings['dataTrain'] = ''
settings['dataValidate'] = ''
settings['epochs'] = 256
settings['gpu'] = '0'
settings['inputSize'] = 128
settings['kappa'] = 1.0
settings['learningRate'] = 0.005
settings['lossFunction'] = 'ce+ed+aw'
settings['model'] = ''
settings['outputArray'] = None
settings['outputDirectory'] = ''
settings['processors'] = multiprocessing.cpu_count()
settings['randomSeed'] = 123456789
settings['remix'] = False
settings['shearFactor'] = 5.0
settings['tau'] = 0.0
settings['weightDecay'] = 0.0001
settings['zoomPercent'] = 10.0



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
weightError = "Class weight dictionary (optional): -w '{0:float,1:float,2:float...}' | --weight='{0:float,1:float,2:float...}'"
try:
	arguments, values = getopt.getopt(sys.argv[1:], 'a:b:cd:e:f:g:hi:K:l:m:o:p:r:s:t:u:v:w:xy:z:', ['array=', 'batch=', 'cpu', 'decay=', 'epochs=', 'function=', 'gpu=', 'help', 'input=', 'kappa=', 'learning=', 'model=', 'output=', 'processors=', 'random=', 'shear=', 'train=', 'tau=', 'validate=', 'weight=', 'cutmix', 'ycount=', 'zoom='])
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
		eprintWrap('A Python3 script to train models on images from .tfr files with TensorFlow 2.9.3.')
		eprintWrap(arrayError)
		eprintWrap(f"Batch size (optional; default = {settings['batch']}): -b int | --batch=int")
		eprintWrap(f"CPU only (optional; default = {not settings['cpu']}): -c | --cpu")
		eprintWrap(f"Weight decay for AdamW optimizer (optional; default = {settings['weightDecay']}): -d float | --decay=float")
		eprintWrap(f"Number of epochs (optional; default = {settings['epochs']}): -e int | --epochs=int")
		eprintWrap(f"Loss, learning rate, and optimization function combination (optional; a = adam; aw = adamW; ce = cross entropy; clr = cyclical learning rate; ed = epoch decay; sgd = stochastic gradient descent; default = {settings['lossFunction']}): -f {'|'.join(LOSSES)} | --function={'|'.join(LOSSES)}")
		eprintWrap(f"Run on specified GPU (optional; default = {settings['gpu']}; CPU option overrides GPU settings): -g int | --gpu=int")
		eprintWrap(f"Input image size (optional; default = {settings['inputSize']}): -i int | --input=int")
		eprintWrap(f"Remix Kappa (optional; default = {settings['kappa']}; arXiv:2007.03943): -K float | --kappa=float")
		eprintWrap(f"Learning rate (optional; default = {settings['learningRate']}): -l float | --learning=float")
		eprintWrap(modelError)
		eprintWrap(outputDirectoryError)
		eprintWrap(f"Processors (optional; default = {settings['processors']}): -p int | --processors=int")
		eprintWrap(f"Random seed (optional; default = {settings['randomSeed']}): -r int | --random=int")
		eprintWrap(f"Shear factor for data augmentation (optional; default = {settings['shearFactor']}): -s float | --shear=float")
		eprintWrap(dataTrainError)
		eprintWrap(f"Remix Tau (optional; default = {settings['tau']}; arXiv:2007.03943): -u float | --tau=float")
		eprintWrap(dataValidateError)
		eprintWrap(weightError)
		# eprintWrap(f"Use cutmix in place of mixup (optional; default = {settings['cutmix']}): -x | --cutmix")
		eprintWrap("Reweigh labels in favor or rare classes using remix (arXiv:2007.03943) using a class count dictionary (optional): -y '{0:int,1:int,2:int...}' | --ycount='{0:int,1:int,2:int...}'")
		eprintWrap(f"Zoom percent for data augmentation (optional; default = {settings['zoomPercent']}): -z float | --zoom=float")
		eprint('')
		sys.exit(0)
	elif argument in ('-i', '--input') and int(value) > 0:
		settings['inputSize'] = int(value)
	elif argument in ('-K', '--kappa') and float(value) > 0.0 and float(value) < 1.0:
		settings['kappa'] = float(value)
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
	elif argument in ('-s', '--shear') and float(value) > 0.0 and float(value) <= 10.0:
		settings['shearFactor'] = float(value)
	elif argument in ('-t', '--train'):
		if os.path.isfile(value):
			settings['dataTrain'] = value
		else:
			eprintWrap(f"Input train file '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-u', '--tau') and float(value) >= 0.0 and float(value) <= 1.0:
		settings['tau'] = float(value)
	elif argument in ('-v', '--validate'):
		if os.path.isfile(value):
			settings['dataValidate'] = value
		else:
			eprintWrap(f"Input validation file '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-w', '--weight'):
		settings['classWeight'] = ast.literal_eval(value)
	# elif argument in ('-x', '--cutmix'):
	# 	settings['cutmix'] = True
	elif argument in ('-y', '--ycount'):
		settings['classCount'] = ast.literal_eval(value)
		settings['remix'] = True
	elif argument in ('-z', '--zoom') and float(value) > 0.0 and float(value) < 100.0:
		settings['optimizer'] = float(value)



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
if not settings['classWeight']:
	settings['classWeight'] = {}
	for k in range(0, settings['outputArray']):
		settings['classWeight'][k] = 1.0
if not settings['classCount']:
	settings['classCount'] = {}
	for k in range(0, settings['outputArray']):
		settings['classCount'][k] = 1

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



### INIT REMIX
if settings['remix']:
	classCounts = tf.cast(tf.constant([settings['classCount'][key] for key in sorted(settings['classCount'].keys())]), tf.int32)



### DATASET FUNCTIONS
def batchMix(images, labels, KAPPA = settings['kappa'], PROBABILITY = 1.0, YCOUNT = settings['classCount']): ### based on https://www.kaggle.com/code/yihdarshieh/batch-implementation-of-more-data-augmentations/notebook?scriptVersionId=29767726 and https://www.kaggle.com/code/cdeotte/cutmix-and-mixup-on-gpu-tpu
	mixup = tf.cast(tf.random.uniform([settings['batch']], 0, 1) <= PROBABILITY, tf.int32)
	mix = tf.random.uniform([settings['batch']], 0, 1)*tf.cast(mixup, tf.float32) ### beta distribution with alpha = 1.0
	new = tf.cast(tf.random.uniform([settings['batch']], 0, settings['batch']), tf.int32)
	if settings['cutmix']:
#
# do not use ->
# 		
		x = tf.cast(tf.random.uniform([settings['batch']], 0, settings['inputSize']), tf.int32)
		y = tf.cast(tf.random.uniform([settings['batch']], 0, settings['inputSize']), tf.int32)
		b = tf.random.uniform([settings['batch']], 0, 1) ### beta distribution with alpha = 1.0
		width = tf.cast(settings['inputSize'] * tf.math.sqrt(1-b), tf.int32) * tf.cast(tf.random.uniform([settings['batch']], 0, 1) <= PROBABILITY, tf.int32)
		width //= 2
		ya = tf.math.maximum(0, y-width)
		yb = tf.math.minimum(settings['inputSize'], y+width)
		xa = tf.math.maximum(0, x-width)
		xb = tf.math.minimum(settings['inputSize'], x+width)
		one = images[:, ya:yb, 0:xa, :]
		two = images[new, ya:yb, xa:xb, :]
		three = images[:, ya:yb, xb:settings['inputSize'], :]
		middle = tf.concat([one, two, three], axis = 1)
		mixupImages = tf.concat([images[:, 0:ya, :, :], middle, images[:, yb:settings['inputSize'], :, :]], axis = 0)
#
# <- do not use
# 		
	else:
		mixupImages =  (1-mix)[:, tf.newaxis, tf.newaxis, tf.newaxis] * images + mix[:, tf.newaxis, tf.newaxis, tf.newaxis] * tf.gather(images, new)
	if settings['remix']:
		decodedLabels = tf.math.argmax(labels, axis = -1)
		decodedNewLabels = tf.math.argmax(tf.gather(labels, new), axis = -1)
		remixRatio = tf.gather(classCounts, decodedLabels)/tf.gather(classCounts, decodedNewLabels)
		newMix = tf.where(tf.math.logical_and(tf.greater_equal(remixRatio, settings['kappa']), tf.less(mix, settings['tau'])), 0.0, mix)
		newMix = tf.where(tf.math.logical_and(tf.less_equal(remixRatio, 1/settings['kappa']), tf.less(1-mix, settings['tau'])), 1.0, newMix)
		mix = newMix
	mixupLabels =  (1-mix)[:, tf.newaxis] * labels + mix[:, tf.newaxis] * tf.gather(labels, new)
	return mixupImages, mixupLabels

def decodeTFR(record):
	feature = {
		'image': tf.io.FixedLenFeature([], tf.string),
		'category': tf.io.FixedLenFeature([], tf.int64)
	}
	record = tf.io.parse_single_example(record, feature)
	record['category'] = tf.one_hot(
		depth = settings['outputArray'],
		indices = record['category']
	)
	record['image'] = tf.cast(tf.io.decode_jpeg(
		channels = 3,
		contents = record['image']
	), tf.float32)
	return record['image'], record['category']

def transform(image, label): ### based on https://www.kaggle.com/code/cdeotte/rotation-augmentation-gpu-tpu-0-96/notebook does not work with odd numbers...
	HALFINPUT = settings['inputSize']//2
	REMAINDER = settings['inputSize']%2
	transform = transformMatrix(
		shear = settings['shearFactor']*tf.random.normal([1], dtype = tf.float32), 
		zoomHeight = 1.0+tf.random.normal([1], dtype = tf.float32)/settings['zoomPercent'], 
		zoomWidth = 1.0+tf.random.normal([1], dtype = tf.float32)/settings['zoomPercent']
	)
	### destination pixel indices
	x = tf.repeat(tf.range(HALFINPUT, -HALFINPUT, -1), settings['inputSize'])
	y = tf.tile(tf.range(-HALFINPUT, HALFINPUT), [settings['inputSize']])
	z = tf.ones([settings['inputSize']*settings['inputSize']], dtype = tf.int32)
	destinationIndex = tf.stack([x, y, z])
	### rotate destination onto origin pixels
	rotateIndex = K.dot(transform, tf.cast(destinationIndex, dtype = tf.float32))
	rotateIndex = tf.cast(rotateIndex, dtype = tf.int32)
	rotateIndex = tf.clip_by_value(rotateIndex, -HALFINPUT+REMAINDER+1, HALFINPUT)
	### find origin pixel values
	originIndex = tf.stack([HALFINPUT-rotateIndex[0,], HALFINPUT-1+rotateIndex[1,]])
	d = tf.gather_nd(image, tf.transpose(originIndex))
	return tf.reshape(d, [settings['inputSize'], settings['inputSize'], 3]), label

def transformMatrix(shear, zoomHeight, zoomWidth): ### returns a 3x3 transform matrix to transform indices; based on https://www.kaggle.com/code/cdeotte/rotation-augmentation-gpu-tpu-0-96/notebook
	ONES = tf.constant([1], dtype = tf.float32)
	SHAPE = (3, 3)
	ZEROS = tf.constant([0], dtype = tf.float32)
	shear = math.pi*shear/180.0
	shearMatrix = tf.reshape(tf.concat([ONES, tf.math.sin(shear), ZEROS, ZEROS, tf.math.cos(shear), ZEROS, ZEROS, ZEROS, ONES], axis = 0), SHAPE)    
	zoomMatrix = tf.reshape(tf.concat([ONES/zoomHeight, ZEROS, ZEROS, ZEROS, ONES/zoomWidth, ZEROS, ZEROS, ZEROS, ONES], axis = 0), SHAPE)
	return K.dot(shearMatrix, zoomMatrix)



### DATASETS
validationData = (
	tf.data.TFRecordDataset(settings['dataValidate'])
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
	tf.data.TFRecordDataset(settings['dataTrain'])
	.map(
		decodeTFR,
		deterministic = False,
		num_parallel_calls = tf.data.AUTOTUNE
	).map(
		lambda x, y: (transform(x, y)),
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
	).map(
		lambda x, y: (batchMix(x, y)),
		deterministic = False,
		num_parallel_calls = tf.data.AUTOTUNE
	).prefetch(tf.data.AUTOTUNE)
)



settings['activation'] = 'selu'
settings['initializer'] = 'glorot_uniform'

### READ AND TRAIN MODEL
def epoch2decayLR(epoch, learningRate):
	return learningRate*1.0/(1.0+(settings['learningRate']/settings['epochs'])*epoch)

model = tf.keras.models.load_model(settings['model'], compile = False)
modelTruncated = tf.keras.Model(inputs = model.input, outputs = model.layers[-2].output)
output = tf.keras.layers.Dense(
	activation = settings['activation'],
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

### loss
loss = None
if settings['lossFunction'] == 'ce+clr+sgd':
	loss = 'categorical_crossentropy'
else:
	loss = tf.keras.losses.CategoricalCrossentropy(
		from_logits = True, 
		label_smoothing = 0.1
	)

### metrics
metrics = [
	tf.keras.metrics.CategoricalAccuracy(
		name = 'accuracy'
	),
	tf.keras.metrics.AUC(
		curve = 'PR',
		from_logits = True,
		multi_label = False, 
		name = 'auc',
		num_thresholds = 200, 
		summation_method = 'interpolation'
	),
	tfa.metrics.F1Score(
		average = 'macro',
		name = 'f1',
		num_classes = settings['outputArray'],
		threshold = None
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
if settings['lossFunction'] in ('ce+clr+a', 'n+clr+a'):
	optimizer = tf.keras.optimizers.Adam(
		learning_rate = clr
	)
elif settings['lossFunction'] in ('ce+clr+aw', 'n+clr+aw'):
	optimizer = tfa.optimizers.AdamW(
		learning_rate = clr,
		weight_decay = settings['weightDecay']
	)
elif settings['lossFunction'] == 'ce+ed+aw':
	optimizer = tfa.optimizers.AdamW(
		learning_rate = settings['learningRate'], 
		weight_decay = settings['weightDecay']
	)
elif settings['lossFunction'] in ('ce+clr+sgd', 'n+clr+sgd'):
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
	mode = 'max',
	monitor = 'val_auc',
	patience = settings['leniency'],
	restore_best_weights = True,
	verbose = 0
))
if settings['lossFunction'] == 'ce+ed+aw':
	callbacks.append(tf.keras.callbacks.LearningRateScheduler(epoch2decayLR))

### train
history = newModel.fit(
	batch_size = settings['batch'],
	callbacks = callbacks,
	class_weight = settings['classWeight'],
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
