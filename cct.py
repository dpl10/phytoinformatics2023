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
CHANNELS = 2
TOKENS = 1
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
settings['encoderDepth'] = 16
settings['encoderUnits'] = 48
settings['mlpBottleneck'] = False
settings['outFile'] = ''
settings['outputArray'] = None
settings['poolTransformers'] = []
settings['randomSeed'] = 123456789
settings['selfAttentionHeads'] = 1
settings['transformerLayers'] = 2
settings['transformerModules'] = 4
settings['vocabulary'] = 10



### OTHER SETTINGS
settings['dformat'] = 'channels_last'
settings['dropout'] = 0.2
settings['epsilon'] = 1e-5
settings['initializer'] = 'glorot_uniform'
settings['positionalEncodingSTD'] = 0.55
settings['randomMax'] = 2**32 ### 64 is unsafe (53 is max safe)
settings['randomMin'] = 0
settings['recodeNoise'] = 1.75
settings['weightDecay'] = 1e-4



### READ OPTIONS
arrayError = 'Number of elements in the output array (required): -a int | --array=int'
outFileError = 'Output file (required): -o file.h5 | --output=file.h5'
try:
	arguments, values = getopt.getopt(sys.argv[1:], 'a:bd:f:hl:m:o:p:r:s:u:v:', ['array=', 'bottleneck', 'depth=', 'function=', 'help', 'layers=', 'modules=', 'output=', 'pool=', 'random=', 'self=', 'units=', 'vocabulary='])
except getopt.error as err:
	eprintWrap(str(err))
	sys.exit(2)
for argument, value in arguments:
	if argument in ('-a', '--array') and int(value) > 0:
		settings['outputArray'] = int(value)
	elif argument in ('-b', '--bottleneck'):
		settings['mlpBottleneck'] = True
	elif argument in ('-d', '--depth') and int(value) > 0 and int(value)%2 == 0:
		settings['encoderDepth'] = int(value)
	elif argument in ('-f', '--function') and value in ACTIVATIONS:
		settings['activation'] = value
	elif argument in ('-h', '--help'):
		eprint('')
		eprintWrap('A Python3 script to create a TensorFlow 2.12.0 Compact Convolutional Transformer-style (CCT; arXiv:2104.05704) model.')
		eprintWrap(arrayError)
		eprintWrap(f"Use ViT style MLP inverted bottleneck (optional; default = {settings['mlpBottleneck']}): -b | --bottleneck")
		eprintWrap(f"Encoder depth (optional; must be an even number; default = {settings['encoderDepth']}): -d int | --depth=int")
		eprintWrap(f"Internal activation function (optional; default = {settings['activation']}): -f {'|'.join(ACTIVATIONS)} | --function={'|'.join(ACTIVATIONS)}")
		eprintWrap('Comma separated list of zero indexed transformer units that use self-calibrated pixel attention in place of conventional softmax attention (optional; default = None; arXiv:2010.01073): -k int,int,int | --calibrated=int,int,int')
		eprintWrap(f"Number of transformer layers (optional; default = {settings['transformerLayers']}): -l int | --layers=int")
		eprintWrap(f"Number of transformer modules (optional; default = {settings['transformerModules']}): -m int | --modules int")
		eprintWrap(outFileError)
		eprintWrap('Comma separated list of zero indexed transformer units that use averagepool attention in place of conventional softmax attention (optional; default = None; arXiv:1706.03762v5 and arXiv:2111.11418): -p int,int,int | --pool=int,int,int')
		eprintWrap(f"Random seed (optional; default = {settings['randomSeed']}): -r int | --random=int")
		eprintWrap(f"Number of heads in the self-attention layer (optional; default = {settings['selfAttentionHeads']}): -s int | --self=int")
		eprintWrap(f"Encoder units (optional; default = {settings['encoderUnits'] }): -u int | --units=int")
		eprintWrap(f"Number of bits needed to capture the vocabulary (optional; default = {settings['vocabulary']}): -v int | --vocabular=int")
		eprint('')
		sys.exit(0)
	elif argument in ('-l', '--layers') and int(value) > 0:
		settings['transformerLayers'] = int(value)
	elif argument in ('-m', '--modules') and int(value) > 0:
		settings['transformerModules'] = int(value)
	elif argument in ('-o', '--output'):
		settings['outFile'] = value
	elif argument in ('-p', '--pool'):
		settings['poolTransformers'] = [int(x) for x in value.split(',')] 
	elif argument in ('-r', '--random') and int(value) >= settings['randomMin'] and int(value) <= settings['randomMax']:
		settings['randomSeed'] = int(value)
	elif argument in ('-s', '--self') and int(value) > 0:
		settings['selfAttentionHeads'] = int(value)
	elif argument in ('-u', '--units') and int(value) > 0:
		settings['encoderUnits'] = int(value)
	elif argument in ('-v', '--vocabulary') and int(value) > 0:
		settings['vocabulary'] = int(value)



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



### AVERAGEPOOL ATTENTION (CF. ARXIV:2111.11418)
def averagepoolAttention(x, name):
	x = tf.keras.layers.Activation(
		'softmax',
		name = f"{name}_activation"
	)(x)
	x = tf.keras.layers.AveragePooling1D(
		data_format = settings['dformat'],
		name = f"{name}_averagepool",
		padding = 'same',
		pool_size = 3,
		strides = 1
	)(x)
	return x

### CCT MODEL (CORRECTED AND MODIFIED FROM https://github.com/keras-team/keras-io/blob/master/examples/vision/cct.py)
def cct():
	input = tf.keras.layers.Input(
		(None, ),
		dtype = tf.int64,
		name = 'input'
	)
	cct = input
	divisor = 2**(settings['vocabulary']//2)
	quotient = embedding(
		x = cct//divisor, 
		encoderDepth = settings['encoderDepth']//2,
		encoderUnits = settings['encoderUnits'] ,
		name = 'quotient',
		vocabularySize = divisor
	)
	remainder = embedding(
		x = cct%divisor, 
		encoderDepth = settings['encoderDepth']//2,
		encoderUnits = settings['encoderUnits'] ,
		name = 'remainder',
		vocabularySize = divisor
	)
	cct = tf.concat((quotient, remainder), axis = -1)
	positionalEncoding = tf.random.truncated_normal(
		shape = (settings['encoderUnits'] , settings['encoderDepth']),
		mean = 0.0,
		stddev = settings['positionalEncodingSTD'],
		dtype = tf.float32,
		seed = random.randint(settings['randomMin'], settings['randomMax']),
		name = 'learnable_positional_encoding'
	)
	cct = tf.math.add(cct, positionalEncoding)
	cct = dense(
		x = cct, 
		activation = settings['activation'], 
		bias = False, 
		name = 'embedding', 
		units = cct.shape.as_list()[-1], 
	)
	cct = tf.keras.layers.Dropout(settings['dropout'])(cct)
	for k in range(settings['transformerModules']):
		cct = transformer(
			x = cct, 
			name = f"transformer_{k}", 
			type = 'averagepool' if k in settings['poolTransformers'] else 'conventional'
		)
	outputAttention = dense(
		x = cct, 
		activation = 'softmax', 
		bias = False, 
		name = 'output_attention', 
		units = 1
	)
	cct *= outputAttention
	cct = tf.keras.layers.GlobalAveragePooling1D(
		data_format = settings['dformat'], 
		keepdims = False,
		name = 'output_gap', 
	)(cct)
	output = dense(
		x = cct, 
		activation = 'tanh', 
		bias = True, 
		name = 'output', 
		units = settings['outputArray']
	)
	return tf.keras.Model(inputs = input, outputs = output)

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

### EMBEDDING
def embedding(x, encoderDepth, encoderUnits, name, vocabularySize):
	return tf.keras.layers.Embedding(
		input_dim = vocabularySize,
		output_dim = encoderDepth,
		embeddings_initializer = settings['initializer'],
		embeddings_regularizer = settings['regularizer'],
		embeddings_constraint = None,
		mask_zero = True,
		input_length = encoderUnits,
		name = f"{name}_embedding" 
	)(x)

### LAYER NORMALIZATION
def normalize(x, name):
	return tf.keras.layers.LayerNormalization(
		axis = -1,
		beta_constraint = None,
		beta_initializer = 'zeros',
		beta_regularizer = None,
		center = True,
		epsilon = settings['epsilon'],
		gamma_constraint = None,
		gamma_initializer = 'ones',
		gamma_regularizer = None,
		name = f"{name}_layerNormalization",
		scale = True
	)(x)

### LINEAR
def linear(x, depth, name):
	x = dense(
		x = x,
		activation = None,
		bias = False,
		name = name,
		units = depth
	)
	x = tf.keras.layers.Dropout(settings['dropout'])(x)
	return x

### MULTI-LAYER PERCEPTRON ENCODER
def mlpEncode(x, bottleneck, name):
	x = normalize(x, f"{name}_mlp")
	units = x.shape.as_list()[CHANNELS]
	for k in range(settings['transformerLayers']):
		x = dense(
			x = x,
			activation = settings['activation'],
			bias = False,
			name = f"{name}_mlp_{k}",
			units = 2*units if bottleneck and k == 0 else units 
		)
		x = tf.keras.layers.Dropout(settings['dropout'])(x)
	x = tf.keras.layers.GaussianNoise( ### active only in training
		seed = random.randint(settings['randomMin'], settings['randomMax']),
		stddev = settings['recodeNoise']
	)(x)
	return x

### SCALED DOT PRODUCT
def scaledDotProduct(q, k, v, name):
	qk = tf.matmul(
		a = q,
		b = k,
		name = f"{name}_qk_matrk_multiply",
		transpose_b = True
	)
	scaled = tf.math.divide(x = qk, y = tf.math.sqrt(tf.cast(k.shape.as_list()[CHANNELS], dtype = tf.float32)))
	return tf.matmul(
		a = tf.nn.softmax(scaled, axis = -1),
		b = v,
		name = f"{name}_qkv_matrk_multiply"
	)

### CLASSIC SELF ATTENTION (ARXIV:1706.03762v5)
def selfAttention(x, heads, name):
	x = normalize(x, name)
	projection = x.shape.as_list()[CHANNELS]
	depth = projection//heads
	attention = []
	weighted = {'q': [], 'k': [], 'v': []}
	for head in range(heads):
		for qkv in weighted.keys():
			weighted[qkv].append(
				linear(
					x = x, 
					depth = depth, 
					name = f"{name}_head_{head}_{qkv}"
				)
			)
		attention.append(scaledDotProduct(weighted['q'][head], weighted['k'][head], weighted['v'][head], name))
	x = tf.concat(attention, axis = -1)
	x = dense(
		x = x,
		activation = None,
		bias = False,
		name = f"{name}_attention",
		units = projection
	)
	return x

### TRANSFORMER UNIT (MODIFIED FROM CCT)
def transformer(x, name, type):
	if type == 'averagepool':
		attention = averagepoolAttention(
			x = x,
			name = f"{name}_attention"
		)
	elif type == 'conventional':
		attention = selfAttention(
			x = x,
			heads = settings['selfAttentionHeads'],
			name = f"{name}_attention"
		)
	x = tf.keras.layers.Add()([attention, x])
	recoder = mlpEncode(
		x = x, 
		bottleneck = settings['mlpBottleneck'],
		name = name
	)
	x = tf.keras.layers.Add()([x, recoder])
	return x



### OUTPUT
model = cct()
eprint(model.summary())
model.save(filepath = settings['outFile'], save_format = 'h5')
sys.exit(0)
