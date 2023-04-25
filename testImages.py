#!/usr/bin/env python3

### SAFE AND REQUIRED IMPORTS
import ast
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
settings['classWeight'] = None
settings['cpu'] = False
settings['dataTest'] = ''
settings['gpu'] = '0'
settings['inputSize'] = 64
settings['model'] = ''
settings['outputArray'] = None
settings['processors'] = multiprocessing.cpu_count()



### READ OPTIONS
arrayError = 'Number of elements in the output array (required): -a int | --array=int'
dataTestError = 'Input test data (required): -t file.tfr | --test file.tfr'
modelError = 'Input model file (required): -m file.h5 | --model=file.h5'
weightError = "Class weight dictionary (required): -w '{0:float,1:float,2:float...}' | --weight='{0:float,1:float,2:float...}'"
try:
	arguments, values = getopt.getopt(sys.argv[1:], 'a:b:cg:hi:m:p:t:w:', ['array=', 'batch=', 'cpu', 'gpu=', 'help', 'input=', 'model=', 'processors=', 'test=', 'weight='])
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
		eprintWrap('A Python3 script to test models on images from .tfr files with TensorFlow 2.9.3.')
		eprintWrap(arrayError)
		eprintWrap(f"Batch size (optional; default = {settings['batch']}): -b int | --batch=int")
		eprintWrap(f"CPU only (optional; default = {not settings['cpu']}): -c | --cpu")
		eprintWrap(f"Run on specified GPU (optional; default = {settings['gpu']}; CPU option overrides GPU settings): -g int | --gpu int")
		eprintWrap(f"Input image size (optional; default = {settings['inputSize']}): -i int | --input=int")
		eprintWrap(modelError)
		eprintWrap(f"Processors (optional; default = {settings['processors']}): -p int | --processors=int")
		eprintWrap(dataTestError)
		eprintWrap(weightError)
		eprint('')
		sys.exit(0)
	elif argument in ('-i', '--input') and int(value) > 0:
		settings['inputSize'] = int(value)
	elif argument in ('-m', '--model'):
		if os.path.isfile(value) or os.path.isdir(value):
			settings['model'] = value
		else:
			eprintWrap(f"Model file '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-p', '--processors') and int(value) > 0:
		settings['processors'] = int(value)
	elif argument in ('-t', '--test'):
		if os.path.isfile(value):
			settings['dataTest'] = value
		else:
			eprintWrap(f"Input file '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-w', '--weight'):
		settings['classWeight'] = ast.literal_eval(value)



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
if not settings['classWeight']:
	settings['classWeight'] = {}
	for k in range(0, settings['outputArray']):
		settings['classWeight'][k] = 1.0

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
eprintWrap(f"TensorFlow GPUs = {len(tf.config.experimental.list_physical_devices('GPU'))}")
eprintWrap(f"TensorFlow {tf.version.VERSION}\n")



### DATASET FUNCTION
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



### DATASET
testData = (
	tf.data.TFRecordDataset(settings['dataTest'])
	.map(
		decodeTFR,
		deterministic = False,
		num_parallel_calls = tf.data.AUTOTUNE
	).batch(
		batch_size = settings['batch'],
		deterministic = False,
		drop_remainder = True,
		# drop_remainder = False,
		num_parallel_calls = tf.data.AUTOTUNE
	).prefetch(tf.data.AUTOTUNE)
)



### READ AND TEST
model = tf.keras.models.load_model(settings['model'], compile = False)
model.compile(
	loss  = tf.keras.losses.CategoricalCrossentropy(
		from_logits = True, 
		label_smoothing = 0.1
	),
	metrics=[
		tf.keras.metrics.CategoricalAccuracy(name = 'accuracy'),
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
	],
	optimizer = tfa.optimizers.AdamW(
		learning_rate = 0.01, 
		weight_decay = 0.001
	)
)
print(model.summary())
stats = model.evaluate(testData)
print(f"Test loss: {stats[0]:.4f}")
print(f"Test accuracy: {(stats[1]*100):.2f}%")
print(f"Test AUCPR: {(stats[2]*100):.2f}%")
print(f"Test macro F1: {(stats[3]*100):.2f}%")
sys.exit(0)
