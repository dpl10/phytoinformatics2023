#!/usr/bin/env python3

### REQUIRED IMPORTS
import ftfy
import getopt
import os
import re
import shutil
import sys
import textwrap



### CONSTANTS
AA = re.compile('[^ACDEFGHIKLMNPQRSTVWYX\-]')
DECIMALS = 2
DNA = re.compile('[^ACGTNVDBHWMRKSY\-]')
DEFLINE = re.compile('^>')
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
settings['aa'] = False
settings['compatible'] = 0
settings['fasta'] = ''
settings['incompatible'] = 1



### READ OPTIONS
fastaError = 'FASTA format alignment to be summarized (required): -f file.fasta | --fasta=file.fasta'
try:
	arguments, values = getopt.getopt(sys.argv[1:], 'ac:f:hi:', ['aa', 'compatible=', 'fasta=', 'help', 'incompatible='])
except getopt.error as err:
	eprintWrap(str(err))
	sys.exit(2)
for argument, value in arguments:
	if argument in ('-a', '--aa'):
		settings['aa'] = True
	elif argument in ('-c', '--compatible') and len(value):
		try:
			settings['compatible'] = int(value)
		except:
			eprintWrap(f"Cannot convert '{value}' to integer!")
			sys.exit(2)
	elif argument in ('-f', '--fasta'):
		if os.path.isfile(value):
			settings['fasta'] = value
		else:
			eprintWrap(f"FASTA file '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-h', '--help'):
		eprintWrap('A Python3 script for computing sum of pairs from a FASTA format alignment.')
		eprintWrap('To output an alignment cost function value, compatible (-c) should be zero and incompatible (-i) should be a positive integer.')
		# eprintWrap('To output a similarity value, compatible (-c) should be a positive integer and incompatible (-i) should be a negative integer.')
		eprintWrap(f"Compute sum of pairs for amino acid alignment (default = {'AA' if settings['aa'] else 'DNA'}): -a | --aa")
		eprintWrap(f"Integer value added for compatible residues (default = {settings['compatible']}): -c int | --compatible=int")
		eprintWrap(fastaError)
		eprintWrap(f"Integer value added for incompatible residues (default = {settings['incompatible']}): -i int | --incompatible=int")
		sys.exit(0)
	elif argument in ('-i', '--incompatible') and len(value):
		try:
			settings['incompatible'] = int(value)
		except:
			eprintWrap(f"Cannot convert '{value}' to integer!")
			sys.exit(2)



### START/END
if not settings['fasta']:
	eprintWrap(fastaError)
	sys.exit(2)
else:
	eprintWrap('Started...')
	for key, value in settings.items():
		eprintWrap(f"{key} = {value}")



### READ AND CLEAN INPUT
def store(alignment, expected, definition, sequence):
	if settings['aa']:
		sequence = re.sub(AA, '', sequence.upper())
	else:
		sequence = re.sub(DNA, '', sequence.upper())
	if len(definition) and len(sequence):
		sequence = list(sequence)
		sequenceLength = len(sequence) if expected else 0
		if expected == sequenceLength:
			tupleSequence = []
			if settings['aa']:
				for residue in sequence:
					if residue == 'X':
						tupleSequence.append(('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'))
					else:
						tupleSequence.append((residue, ))
			else:
				for residue in sequence:
					if residue in ('A', 'C', 'G', 'T', '-'):
						tupleSequence.append((residue, ))
					elif residue == 'B':
						tupleSequence.append(('C', 'G', 'T'))
					elif residue == 'D':
						tupleSequence.append(('A', 'G', 'T'))
					elif residue == 'H':
						tupleSequence.append(('A', 'C', 'T'))
					elif residue == 'K':
						tupleSequence.append(('G', 'T'))
					elif residue == 'M':
						tupleSequence.append(('A', 'C'))
					elif residue == 'N':
						tupleSequence.append(('A', 'C', 'G', 'T'))
					elif residue == 'R':
						tupleSequence.append(('A', 'G'))
					elif residue == 'S':
						tupleSequence.append(('G', 'C'))
					elif residue == 'V':
						tupleSequence.append(('A', 'C', 'G'))
					elif residue == 'W':
						tupleSequence.append(('A', 'T'))
					elif residue == 'Y':
						tupleSequence.append(('C', 'T'))
			alignment.append(tupleSequence)
			return len(sequence)
		else:
			eprintWrap(f"All sequences must be the same length! Previous sequences = {expected}; '{definition}' = {sequenceLength}")
			sys.exit(2)
	else:
		return 0

alignment = []
definition = ''
expected = 0
sequence = ''
with open(settings['fasta'], mode = 'rt', encoding = 'utf8', errors = 'replace') as file:
	for line in file:
		text = ftfy.fix_text(line).strip()
		if re.search(DEFLINE, text):
			expected = store(alignment, expected, definition, sequence)
			definition = text
			sequence = ''
		else:
			sequence += text
expected = store(alignment, expected, definition, sequence)



### COMPUTE SUM OF PAIRS
# increment = max(settings['compatible'], settings['incompatible'])
sum = 0
# total = 0 
for k in range(0, len(alignment)-1):
	for j in range(k+1, len(alignment)):
		for i in range(0, expected):
			# total += increment
			if any(e in alignment[k][i] for e in alignment[j][i]):
				sum += settings['compatible']
			else:
				sum += settings['incompatible']



### OUTPUT
print('')
print(f"Sum of pairs = {sum:,}")
# print(f"Maximum cost = {total:,}")
# print(f"Percent identity = {(100*(sum/total)):.{DECIMALS}f}")