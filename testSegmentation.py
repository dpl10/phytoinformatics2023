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
settings['gpu'] = '0'
settings['inputSize'] = 128
settings['model'] = ''
settings['outputArray'] = None
settings['processors'] = multiprocessing.cpu_count()



### READ OPTIONS
arrayError = 'Number of elements in the output array (required): -a int | --array=int'
dataTestError = 'Input test data (required): -t file.tfr | --test file.tfr'
modelError = 'Input model file (required): -m file.h5 | --model=file.h5'
try:
	arguments, values = getopt.getopt(sys.argv[1:], 'a:b:cg:hi:m:p:t:', ['array=', 'batch=', 'cpu', 'gpu=', 'help', 'input=', 'model=', 'processors=', 'test='])
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
eprintWrap(f"TensorFlow GPUs = {len(tf.config.experimental.list_physical_devices('GPU'))}")
eprintWrap(f"TensorFlow {tf.version.VERSION}\n")



### DATASET FUNCTION
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



### DATASET
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
		# drop_remainder = False,
		num_parallel_calls = tf.data.AUTOTUNE
	).prefetch(tf.data.AUTOTUNE)
)



### READ AND TEST
model = tf.keras.models.load_model(settings['model'], compile = False)
model.compile(
	loss = tf.keras.losses.SparseCategoricalCrossentropy(
		from_logits = True
	),
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
print(f"Test MeanIoU: {(stats[2]*100):.2f}%")
sys.exit(0)
