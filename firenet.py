#!/usr/bin/env python3

### SAFE IMPORTS
import getopt
import os
import random
import shutil
import sys
import textwrap



### CONSTANTS
ACTIVATIONS = ('elu', 'gelu', 'relu', 'selu', 'swish')
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
settings['activation'] = 'selu'
settings['depthwise'] = False
settings['inputSize'] = 128
settings['outFile'] = ''
settings['outputArray'] = None
settings['randomSeed'] = 123456789



### OTHER SETTINGS
settings['dformat'] = 'channels_last'
settings['dropout'] = 0.2
settings['initializer'] = 'glorot_uniform'
settings['randomMax'] = 2**32 ### 64 is unsafe (53 is max safe)
settings['randomMin'] = 0
settings['weightDecay'] = 1e-4



### READ OPTIONS
arrayError = 'Number of elements in the output array (required): -a int | --array=int'
outFileError = 'Output file (required): -o file.h5 | --output=file.h5'
try:
	arguments, values = getopt.getopt(sys.argv[1:], 'a:df:hi:o:r:', ['array=', 'depthwise', 'function=', 'help', 'input=', 'output=', 'random='])
except getopt.error as err:
	eprintWrap(str(err))
	sys.exit(2)
for argument, value in arguments:
	if argument in ('-a', '--array') and int(value) > 0:
		settings['outputArray'] = int(value)
	elif argument in ('-d', '--depthwise'):
		settings['depthwise'] = True
	elif argument in ('-f', '--function') and value in ACTIVATIONS:
		settings['activation'] = value
	elif argument in ('-h', '--help'):
		eprint('')
		eprintWrap('A Python3 script to create a TensorFlow 2.9.3 reduced FireNet (DOI:10.1007/S11063-021-10555-1) model.')
		eprintWrap(arrayError)
		eprintWrap(f"Modify (arXiv:1907.02157) Fire Module (arXiv:1602.07360) 3x3 convolution to use depth-wise convolution (optional; default = {settings['depthwise']}): -d | -- depthwise")
		eprintWrap(f"Internal activation function (optional; default = {settings['activation']}): -f {'|'.join(ACTIVATIONS)} | --function={'|'.join(ACTIVATIONS)}")
		eprintWrap(f"Input image size (optional; default = {settings['inputSize']}): -i int | --input=int")
		eprintWrap(outFileError)
		eprintWrap(f"Random seed (optional; default = {settings['randomSeed']}): -r int | --random=int")
		eprint('')
		sys.exit(0)
	elif argument in ('-i', '--input') and int(value) > 0:
		settings['inputSize'] = int(value)
	elif argument in ('-o', '--output'):
		settings['outFile'] = value
	elif argument in ('-r', '--random') and int(value) >= settings['randomMin'] and int(value) <= settings['randomMax']:
		settings['randomSeed'] = int(value)



### START/END
if not settings['outputArray']:
	eprintWrap(arrayError)
	sys.exit(2)
elif not settings['outFile']:
	eprintWrap(outFileError)
	sys.exit(2)
else:
	eprintWrap('started...')
	for key, value in settings.items():
		eprintWrap(f"{key} = {value}")



### DISABLE GPU, THEN IMPORT TENSORFLOW
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
eprintWrap(f"TensorFlow {tf.version.VERSION}\n")



### INIT
random.seed(settings['randomSeed'])
tf.random.set_seed(random.randint(settings['randomMin'], settings['randomMax']))
settings['regularizer'] = tf.keras.regularizers.L2(
	l2 = settings['weightDecay']
)



### CONV2D
def conv2D(x, activation, dilation, filters, groups, kernel, name, padding, strides):
	return tf.keras.layers.Conv2D(
		activation = activation,
		activity_regularizer = None,
		bias_constraint = None,
		bias_initializer = None,
		bias_regularizer = None,
		data_format = settings['dformat'],
		dilation_rate = dilation,
		filters = filters,
		groups = groups,
		kernel_constraint = None,
		kernel_initializer = settings['initializer'],
		kernel_regularizer = settings['regularizer'],
		kernel_size = kernel,
		name = f"{name}_conv2D",
		padding = padding,
		strides = strides,
		use_bias = False
	)(x)

### DENSE
def dense(x, activation, bias, name, units, zeros = True):
	return tf.keras.layers.Dense(
		activation = activation,
		activity_regularizer = None,
		bias_constraint = None,
		bias_initializer = 'zeros' if zeros else 'ones',
		bias_regularizer = None,
		kernel_constraint = None,
		kernel_initializer = settings['initializer'],
		kernel_regularizer = None,
		name = f"{name}_dense",
		units = units,
		use_bias = bias
	)(x)

### DCONV2D
def dconv2D(x, activation, dilation, kernel, name, padding, strides):
	return tf.keras.layers.DepthwiseConv2D(
		activation = activation,
		activity_regularizer = None,
		bias_constraint = None,
		bias_initializer = None,
		bias_regularizer = None,
		data_format = settings['dformat'],
		depth_multiplier = 1,
		depthwise_constraint = None,
		depthwise_initializer = settings['initializer'],
		depthwise_regularizer = settings['regularizer'],
		dilation_rate = dilation,
		kernel_size = kernel,
		name = f"{name}_dconv2D",
		padding = padding,
		strides = strides,
		use_bias = False,
	)(x)

### FIRE MODULE (ARXIV:1602.07360) WITH OPTIONAL MODIFICATION (ARXIV:1907.02157)
def fireModule(filters, fire, name, reduce):
	fire = conv2D(
		x = fire,
		activation = settings['activation'],
		dilation = 1,
		filters = filters,
		groups = 1,
		kernel = (1, 1),
		name = f"{name}_squeezeFire",
		padding = 'same',
		strides = 1
	)
	if settings['depthwise']:
		bigExpand = dconv2D(
			x = fire,
			activation = settings['activation'],
			dilation = 1,
			kernel = (3, 3),
			name = f"{name}_bigExpandFire",
			padding = 'same',
			strides = 1
		)
	else:
		bigExpand = conv2D(
			x = fire,
			activation = settings['activation'],
			dilation = 1,
			filters = filters,
			groups = 1,
			kernel = (3, 3),
			name = f"{name}_bigExpandFire",
			padding = 'same',
			strides = 1
		)
	smallExpand = conv2D(
		x = fire,
		activation = settings['activation'],
		dilation = 1,
		filters = filters,
		groups = 1,
		kernel = (1, 1),
		name = f"{name}_smallExpandFire",
		padding = 'same',
		strides = 1
	)
	fire = tf.concat(
		axis = -1,
		name = f"{name}_concatFire",
		values = [bigExpand, smallExpand]
	)
	if reduce:
		fire = tf.keras.layers.MaxPool2D(
			name = f"{name}_fire_maxpool2D",
			padding = 'same',
			pool_size = (3, 3),
			strides = 2
		)(fire)
	return fire

### REDUCED FIRENET (DOI:10.1007/S11063-021-10555-1)
def firenet():
	input = tf.keras.layers.Input(
		(None, None, 3),
		name = 'input'
	)
	firenet = input
	firenet = tf.keras.layers.RandomCrop( ### active in both training and inference: random in training; in inference rescaled to preserve the shorter side and center cropped
		height = settings['inputSize'],
		seed = random.randint(settings['randomMin'], settings['randomMax']),
		width = settings['inputSize']
	)(firenet)
	firenet = tf.keras.layers.Rescaling(
		name = 'rescale',
		offset = -1,
		scale = 1.0/127.5 
	)(firenet)
	firenet = conv2D(
		x = firenet,
		activation = settings['activation'],
		dilation = 1,
		filters = 32,
		groups = 1,
		kernel = (3, 3),
		name = 'fire0',
		padding = 'same',
		strides = 1
	)
	firenet = fireModule(
		filters = 16, 
		fire = firenet, 
		name = 'fire1', 
		reduce = True
	)
	firenet = fireModule(
		filters = 24, 
		fire = firenet, 
		name = 'fire2', 
		reduce = True
	)
	firenet = tf.keras.layers.Dropout(settings['dropout'])(firenet)
	firenet = fireModule(
		filters = 24, 
		fire = firenet, 
		name = 'fire3', 
		reduce = True
	)
	firenet = fireModule(
		filters = 16, 
		fire = firenet, 
		name = 'fire4', 
		reduce = False
	)
	firenet = tf.keras.layers.GlobalAveragePooling2D(
		data_format = settings['dformat'], 
		keepdims = False,
		name = 'output_gap', 
	)(firenet)
	firenet = dense(
		x = firenet, 
		activation = settings['activation'], 
		bias = True, 
		name = 'gap_resolve', 
		units = firenet.shape.as_list()[1]
	)
	output = dense(
		x = firenet, 
		activation = None, 
		bias = True, 
		name = 'output', 
		units = settings['outputArray']
	)
	return tf.keras.Model(inputs = input, outputs = output)



### OUTPUT
model = firenet()
eprint(model.summary())
model.save(filepath = settings['outFile'], save_format = 'h5')
sys.exit(0)
