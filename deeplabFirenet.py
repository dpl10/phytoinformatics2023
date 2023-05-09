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
settings['interpolation'] = 'bilinear' ### try area (crashes in 2.9.3), bicubic (4000 ms/step), bilinear (180 ms/step), lanczos3 (2000 ms/step), lanczos5, mitchellcubic (2000 ms/step), nearest (180 ms/step)
settings['randomMax'] = 2**32 ### 64 is unsafe (53 is max safe)
settings['randomMin'] = 0
settings['weightDecay'] = 1e-4



### READ OPTIONS
arrayError = 'Number of elements in the output array (classes; required): -a int | --array=int'
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
		eprintWrap('A Python3 script to create a TensorFlow 2.12.0 reduced FireNet (DOI:10.1007/S11063-021-10555-1) with an attached modified DeepLabv3+ (arXiv:1802.02611) semantic segmentation module.')
		eprintWrap(arrayError)
		eprintWrap(f"Modify (arXiv:1907.02157) Fire Module (arXiv:1602.07360) 3x3 convolution to use depth-wise convolution (optional; default = {settings['depthwise']}): -d | --depthwise")
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
	fire0 = input
	fire0 = tf.keras.layers.RandomCrop( ### active in both training and inference: random in training; in inference rescaled to preserve the shorter side and center cropped
		height = settings['inputSize'],
		seed = random.randint(settings['randomMin'], settings['randomMax']),
		width = settings['inputSize']
	)(fire0)
	fire0 = tf.keras.layers.Rescaling(
		name = 'rescale',
		offset = -1,
		scale = 1.0/127.5 
	)(fire0)
	fire0 = conv2D(
		x = fire0,
		activation = settings['activation'],
		dilation = 1,
		filters = 32,
		groups = 1,
		kernel = (3, 3),
		name = 'fire0',
		padding = 'same',
		strides = 1
	)
	fire1 = fireModule(
		filters = 16, 
		fire = fire0, 
		name = 'fire1', 
		reduce = True
	)
	fire2 = fireModule(
		filters = 24, 
		fire = fire1, 
		name = 'fire2', 
		reduce = True
	)
	fire2 = tf.keras.layers.Dropout(settings['dropout'])(fire2)
	fire3 = fireModule(
		filters = 24, 
		fire = fire2, 
		name = 'fire3', 
		reduce = True
	)
	fire4 = fireModule(
		filters = 16, 
		fire = fire3, 
		name = 'fire4', 
		reduce = False
	)
	pyramid = deepLab3Pyramid(
		image = fire4, 
		name = 'deepLab4', 
		outputChannels = fire4.shape.as_list()[3]//2, 
		scale = 1, 
		upsample = fire1.shape.as_list()[1]//fire4.shape.as_list()[1]
	)
	decoder = conv2D(
		x = fire1,
		activation = settings['activation'],
		dilation = 1,
		filters = fire4.shape.as_list()[3]//8,
		groups = 1,
		kernel = (1, 1),
		name = 'decoder_bottleneck',
		padding = 'same',
		strides = 1
	)
	decoder = tf.concat(
		axis = -1,
		name = 'decoder_concat',
		values = [pyramid, decoder]
	)
	decoder = conv2D(
		x = decoder,
		activation = settings['activation'],
		dilation = 1,
		filters = decoder.shape.as_list()[3],
		groups = 1,
		kernel = (3, 3),
		name = 'decoder_pyramid_resolve',
		padding = 'same',
		strides = 1
	)
	decoder = conv2D(
		x = decoder,
		activation = settings['activation'],
		dilation = 1,
		filters = fire4.shape.as_list()[3],
		groups = 1,
		kernel = (3, 3),
		name = 'decoder_smooth',
		padding = 'same',
		strides = 1
	)
	decoder = tf.keras.layers.UpSampling2D(
		data_format = settings['dformat'], 
		interpolation = settings['interpolation'],
		name = 'decoder_upSample',
		size = (fire0.shape.as_list()[1]//decoder.shape.as_list()[1], fire0.shape.as_list()[2]//decoder.shape.as_list()[2])
	)(decoder)
	output = conv2D(
		x = decoder,
		activation = None,
		dilation = 1,
		filters = settings['outputArray'],
		groups = 1,
		kernel = (1, 1),
		name = 'decoder_output',
		padding = 'same',
		strides = 1
	)
	return tf.keras.Model(inputs = input, outputs = output)

def deepLab3Pyramid(image, name, outputChannels, scale, upsample): ### with help from https://keras.io/examples/vision/deeplabv3_plus/
	batch, x, y, channels = image.shape.as_list()
	channels //= 2
	pyramid = []
	pool = tf.keras.layers.AveragePooling2D(
		data_format = settings['dformat'],
		name = f"{name}_pyramid_averagepool",
		padding = 'valid',
		pool_size = (x, y),
		strides = None
	)(image)
	pool = conv2D(
		x = pool,
		activation = settings['activation'],
		dilation = 1,
		filters = channels,
		groups = 1,
		kernel = (1, 1),
		name = f"{name}_pyramid_averagepool",
		padding = 'same',
		strides = 1
	)
	pool = tf.keras.layers.UpSampling2D(
		data_format = settings['dformat'], 
		interpolation = settings['interpolation'],
		name = f"{name}_averapool_upSample",
		size = (x//pool.shape.as_list()[1], y//pool.shape.as_list()[2])
	)(pool)
	pyramid.append(pool)
	pyramid1x1 = conv2D(
		x = image,
		activation = settings['activation'],
		dilation = 1,
		filters = channels,
		groups = 1,
		kernel = (1, 1),
		name = f"{name}_pyramid1x1",
		padding = 'same',
		strides = 1
	)
	pyramid.append(pyramid1x1)
	pyramidSmall = conv2D(
		x = image,
		activation = settings['activation'],
		dilation = 1,
		filters = channels,
		groups = 1,
		kernel = (1, 1),
		name = f"{name}_pyramidSmall1x1",
		padding = 'same',
		strides = 1
	)
	pyramidSmall = conv2D(
		x = pyramidSmall,
		activation = settings['activation'],
		dilation = 2*scale,
		filters = channels,
		groups = 1,
		kernel = (3, 3),
		name = f"{name}_pyramidSmall",
		padding = 'same',
		strides = 1
	)
	pyramid.append(pyramidSmall)
	pyramidMedium = conv2D(
		x = image,
		activation = settings['activation'],
		dilation = 1,
		filters = channels,
		groups = 1,
		kernel = (1, 1),
		name = f"{name}_pyramidMedium1x1",
		padding = 'same',
		strides = 1
	)
	pyramidMedium = conv2D(
		x = pyramidMedium,
		activation = settings['activation'],
		dilation = 4*scale,
		filters = channels,
		groups = 1,
		kernel = (3, 3),
		name = f"{name}_pyramidMedium",
		padding = 'same',
		strides = 1
	)
	pyramid.append(pyramidMedium)
	pyramidLarge = conv2D(
		x = image,
		activation = settings['activation'],
		dilation = 1,
		filters = channels,
		groups = 1,
		kernel = (1, 1),
		name = f"{name}_pyramidLarge1x1",
		padding = 'same',
		strides = 1
	)
	pyramidLarge = conv2D(
		x = pyramidLarge,
		activation = settings['activation'],
		dilation = 8*scale,
		filters = channels,
		groups = 1,
		kernel = (3, 3),
		name = f"{name}_pyramidLarge",
		padding = 'same',
		strides = 1
	)
	pyramid.append(pyramidLarge)
	pyramid = tf.concat(
		axis = -1,
		name = f"{name}_pyramid_concat",
		values = pyramid
	)
	pyramid = conv2D(
		x = pyramid,
		activation = settings['activation'],
		dilation = 1,
		filters = outputChannels,
		groups = 1,
		kernel = (1, 1),
		name = f"{name}_pyramid_resolve",
		padding = 'same',
		strides = 1
	)
	pyramid = tf.keras.layers.UpSampling2D(
		data_format = settings['dformat'], 
		interpolation = settings['interpolation'],
		name = f"{name}_pyramid_upSample",
		size = (upsample, upsample)
	)(pyramid)
	return pyramid



### OUTPUT
model = firenet()
eprint(model.summary())
model.save(filepath = settings['outFile'], save_format = 'h5')
sys.exit(0)
