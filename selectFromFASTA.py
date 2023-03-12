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
CHAR = re.compile('[^ABCDEFGHIKLMNOPQRSTUVWXY\-]')
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
settings['fasta'] = ''
settings['invert'] = False
settings['list'] = ''



### OTHER SETTINGS
settings['columns'] = 80



### READ OPTIONS
fastaError = 'FASTA file to be searched (required): -f file.fasta | --fasta=file.fasta'
listError = 'List of sequence definitions to output, one per line (required): -l file.txt | --list=file.txt'
try:
	arguments, values = getopt.getopt(sys.argv[1:], 'f:hil:', ['fasta=', 'help', 'invert', 'list='])
except getopt.error as err:
	eprintWrap(str(err))
	sys.exit(2)
for argument, value in arguments:
	if argument in ('-f', '--fasta'):
		if os.path.isfile(value):
			settings['fasta'] = value
		else:
			eprintWrap(f"FASTA file '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-h', '--help'):
		eprintWrap('A Python3 script for selecting sequences from a FASTA file. If there are duplicate selected sequences, only the first one is output.')
		eprintWrap(fastaError)
		eprintWrap(f"Invert selections (i.e. omit listed sequences; optional; default = {settings['invert']}): -i | --invert")
		eprintWrap(listError)
		sys.exit(0)
	elif argument in ('-i', '--invert'):
		settings['invert'] = True
	elif argument in ('-l', '--list'):
		if os.path.isfile(value):
			settings['list'] = value
		else:
			eprintWrap(f"List file '{value}' does not exist!")
			sys.exit(2)



### START/END
if not settings['fasta']:
	eprintWrap(fastaError)
	sys.exit(2)
elif not settings['list']:
	eprintWrap(listError)
	sys.exit(2)
else:
	eprintWrap('Started...')
	for key, value in settings.items():
		eprintWrap(f"{key} = {value}")



### FUNCTIONS
def breakLines(sequence):
	output = ''
	for k, base in enumerate(sequence):
		if k > 0 and k % settings['columns'] == 0:
			output += '\n'
		output += base
	return output

def output(definition, definitions, sequence):
	if len(definition) and len(sequence) and ((settings['invert'] == False and definition in definitions and definitions[definition] == False) or (settings['invert'] == True and definition not in definitions)):
		print(definition)
		print(breakLines(re.sub(CHAR, '', sequence.upper())))
		definitions[definition] = True



### LOAD LIST
definitions = {} ### def => bool
with open(settings['list'], mode = 'rt', encoding = 'utf8', errors = 'replace') as file:
	for line in file:
		text = ftfy.fix_text(line).strip()
		if len(text):
			definitions[text] = False



### READ AND OUTPUT
definition = ''
sequence = ''
with open(settings['fasta'], mode = 'rt', encoding = 'utf8', errors = 'replace') as file:
	for line in file:
		text = ftfy.fix_text(line).strip()
		if re.search(DEFLINE, text):
			output(definition, definitions, sequence)
			definition = text
			sequence = ''
		else:
			sequence += text
output(definition, definitions, sequence)
