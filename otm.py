#!/usr/bin/env python3

### IMPORTS
import getopt
import math
import os
import re
import sys
import textwrap

### GLOBAL CONSTANTS
BASE2NUMBER = {
	'A': (0, ),
	'C': (1, ),
	'G': (2, ),
	'T': (3, ),
	'K': (2, 3),
	'M': (0, 1),
	'R': (0, 2),
	'S': (1, 2),
	'W': (0, 3),
	'Y': (1, 3),
	'B': (1, 2, 3),
	'D': (0, 2, 3),
	'H': (0, 1, 3),
	'V': (0, 1, 2),
	'N': (0, 1, 2, 3)
}
BASE2WEIGHT = {
	'B': 0.6666,
	'C': 1.0000,
	'D': 0.3333,
	'G': 1.0000,
	'H': 0.3333,
	'K': 0.5000,
	'M': 0.5000,
	'N': 0.5000,
	'R': 0.5000,
	'S': 1.0000,
	'V': 0.6666,
	'Y': 0.5000
}
DECIMALS = 1
DNA = re.compile('^[ABCDGHKMNRSTVWY]+$', re.IGNORECASE)
DNAPM = re.compile('^[0-5]{0,1}\.{0,1}[0-9]{0,2}$')
MONOMM = re.compile('^[1-9]|[1-9][0-9]|[1-9][0-9]{2,2}|[1-4][0-9]{3,3}|5000$')
OLIGOS = ('left', 'right', 'template')
WRAP = int(os.popen('stty size', 'r').read().split()[1])

### GLOBAL USER SETTINGS AND DEFAULTS
settings = {}
settings['dna'] = 1.0 ### oligo DNA concentration (pM)
settings['left'] = '' ### left oligo sequence
settings['mono'] = 50.0 ### monovalent cation concentration (mM)
settings['right'] = '' ### right oligo sequence
settings['template'] = '' ### template sequence
settings['KB'] = False ### use the Khandelwal & Bhyravabhotla algorithm

### FUNCTIONS
def eprint(*arguments, **keywordArguments):
	print(*arguments, file = sys.stderr, **keywordArguments)

def sequenceError(x):
	return wrap(f'{x.title()} sequence (required): -{x[0]} ACGT... | --{x}=ACGT...')

def wrap(string, columns = WRAP):
	return '\n'.join(textwrap.wrap(string, columns))

def oligoScore(oligo, matrix): ### modified to work with polymorphic bases
	score = 0.0
	for position in range(1, len(oligo)):
		trailingBase = BASE2NUMBER[oligo[position-1]]
		leadingBase = BASE2NUMBER[oligo[position]]
		sum = 0.0
		for trailing in trailingBase:
			for leading in leadingBase:
				sum += matrix[trailing][leading]
		score += sum/(len(trailingBase)*len(leadingBase))
	return score

def KB(oligo, mono = settings['mono'], dna = settings['dna']): ### Khandelwal & Bhyravabhotla
	return 7.35*(oligoScore(oligo, (
		(5.0, 10.0,  8.0,  7.0), ### AA,AC,AG,AT
		(7.0, 11.0, 10.0,  8.0), ### CA,CC,CG,CT
		(8.0, 13.0, 11.0, 10.0), ### GA,GC,GG,GT
		(4.0,  8.0,  7.0,  5.0)  ### TA,TC,TG,TT
	))/len(oligo)) \
	+ 17.34*math.log(len(oligo), 10) \
	+ 4.96*math.log(mono/1000.0, 10) \
	+ 0.89*math.log(dna/1000000000000.0, 10) \
	- 25.42

def RSR(oligo, mono = settings['mono']): ### Rychlik, Spencer, & Rhoads
	dH = oligoScore(oligo, (
		(9.1,  6.5,  7.8,  8.6), ### AA,AC,AG,AT
		(5.8, 11.0, 11.9,  7.8), ### CA,CC,CG,CT
		(5.6, 11.1, 11.0,  6.5), ### GA,GC,GG,GT
		(6.0,  5.6,  5.8,  9.1), ### TA,TC,TG,TT
	))
	dS = oligoScore(oligo, (
		(24.0, 17.3, 20.8, 23.9), ### AA,AC,AG,AT
		(12.9, 26.6, 27.8, 20.8), ### CA,CC,CG,CT
		(13.5, 26.7, 26.6, 17.3), ### GA,GC,GG,GT
		(16.9, 13.5, 12.9, 24.0)  ### TA,TC,TG,TT
	))
	return (-1000.0*dH)/(-1.0*dS - 57.48) \
	- 273.15 \
	+ 16.6*math.log(mono/1000.0, 10)

### READ OPTIONS
try:
	arguments, values = getopt.getopt(
		sys.argv[1:],
		'd:hkl:m:r:t:',
		['dna=', 'help', 'kb', 'left=', 'monovalent=', 'right=', 'template=']
	)
except getopt.error as error:
	eprint(str(error))
	sys.exit(2)
for argument, value in arguments:
	if argument in ('-d', '--dna') and re.search(DNAPM, value) and float(value) > 0.0:
		settings['dna'] = float(value)
	elif argument in ('-h', '--help'):
		eprint('')
		eprint(wrap('A Python3 script for computing optimal PCR annealing temperatures using '
			'either the algorithm of Rychlik et al. [1] with Osborne’s [2] corrections and '
			'additional modifications for polymorphic sequences (thermodynamic constants from '
			'Breslauer et al. [3]) or the algorithm of Khandelwal & Bhyravabhotla [4] modified '
			'for polymorphic sequences using the Rychlik et al. [1] combining formula (without '
			'Osborne’s [2] corrections).'
		))
		eprint('\nREFERENCES:')
		eprint(wrap('[1] Rychlik, W., W.J. Spencer, & R.E. Rhoads. 1990. Optimization of the '
			'annealing temperature for DNA amplification in vitro. Nucleic Acids Research 18: '
			'6409–6412. https://doi.org/10.1093/nar/18.21.6409'
		))
		eprint(wrap('[2] Osborne, B.I. 1992. HyperPCR: a Macintosh Hypercard program for the '
			'determination of optimal PCR annealing temperature. CABIOS 8: 83. '
			'https://doi.org/10.1093/bioinformatics/8.1.83'
		))
		eprint(wrap('[3] Breslauer, K.J., R. Frank, H. Blocker, & L.A. Marky. 1986. Predicting '
			'DNA duplex stability from the base sequence. Proceedings of the National Academy of '
			'Sciences of the United States of America 83: 3746–3750. '
			'https://doi.org/10.1073/pnas.83.11.3746'
		))
		eprint(wrap('[4] Khandelwal, G. & J. Bhyravabhotla. 2010. A phenomenological model for '
			'predicting melting temperatures of DNA sequences. PLOS ONE 5: e12433. '
			'https://doi.org/10.1371/journal.pone.0012433'
		))
		eprint('\nOPTIONS:')
		eprint(wrap('Oligo DNA concentration (pM; optional; default = '
			f"{settings['dna']:.{DECIMALS}f} pM): -d x.x | --dna=x.x"
		))
		eprint(wrap('Use the Khandelwal & Bhyravabhotla [4] algorithm (optional; default = '
			f"{settings['KB']}): -k | --kb"
		))
		eprint(sequenceError('left'))
		eprint(wrap('Monovalent cation concentration (mM; optional; default = '
			f"{settings['mono']:.{DECIMALS}f} mM): -m x.x | --monovalent=x.x"
		))
		eprint(sequenceError('right'))
		eprint(f"{sequenceError('template')}\n")
		sys.exit(0)
	elif argument in ('-k', '--kb'):
		settings['KB'] = True
	elif argument in ('-l', '--left') and re.search(DNA, value):
		settings['left'] = value.upper()
	if argument in ('-m', '--monovalent') and re.search(MONOMM, value) and float(value) > 0.0:
		settings['mono'] = float(value)
	elif argument in ('-r', '--right') and re.search(DNA, value):
		settings['right'] = value.upper()
	elif argument in ('-t', '--template') and re.search(DNA, value):
		settings['template'] = value.upper()

### START OR END
for oligo in OLIGOS:
	if not settings[oligo]:
		eprint(sequenceError(oligo))
		sys.exit(2)

### COMPUTE TM
print('\nOUTPUT:')
algorithm = 'Khandelwal & Bhyravabhotla' if settings['KB'] else 'Rychlik, Spencer, & Rhoads'
print(wrap(f'Tm calculated with a modified {algorithm} algorithm.'))
print(wrap(f"Monovalent cation concentration = {settings['mono']:.{DECIMALS}f} mM."))
print(wrap(f"Oligo DNA concentration = {settings['dna']:.{DECIMALS}f} pM."))

tm = {}
for oligo in OLIGOS[0:-1]:
	tm[oligo] = KB(settings[oligo]) if settings['KB'] else RSR(settings[oligo])
	print(wrap(f"{oligo.title()} oligo Tm = {tm[oligo]:.{DECIMALS}f}°C ({settings[oligo]})."))

gc = 0.0
for base, weight in BASE2WEIGHT.items():
	gc += settings['template'].count(base)*weight
gc /= len(settings['template'])
tm['template'] = 41.0*gc \
+ 16.6*math.log(settings['mono']) \
- 675/len(settings['template']) ### log base e contra Rychlik, Spencer, & Rhoads
print(wrap(f"Template Tm = {tm['template']:.{DECIMALS}f}°C."))

tm['optimal'] = min(tm['left'], tm['right'])
tm['optimal'] += 0.0 if settings['KB'] else 14.0
tm['optimal'] *= 0.3
tm['optimal'] += 0.7*tm['template']
tm['optimal'] -= 14.9 if settings['KB'] else 22.9
print(wrap(f"Optimal combined Tm = {tm['optimal']:.{DECIMALS}f}°C."))
print('')
sys.exit(0)
