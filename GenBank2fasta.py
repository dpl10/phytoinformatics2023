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
INTRA = re.compile('(convar\.|f\.|forma|grex|lusus|microgene|modif\.|monstr\.|mut\.|nothosubsp\.|nothovar\.|proles|provar\.|spp\.|spp|stirps|subf\.|sublusus|subproles|subso|subsp\.|subvar\.|var\.|var)')
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
settings['accessions'] = False
settings['fasta'] = ''



### OTHER SETTINGS
settings['columns'] = 80



### READ OPTIONS
fastaError = 'FASTA file to be cleaned (required): -f file.fasta | --fasta=file.fasta'
try:
	arguments, values = getopt.getopt(sys.argv[1:], 'af:h', ['accessions', 'fasta=', 'help'])
except getopt.error as err:
	eprintWrap(str(err))
	sys.exit(2)
for argument, value in arguments:
	if argument in ('-a', '--accessions'):
		settings['accessions'] = True
	elif argument in ('-f', '--fasta'):
		if os.path.isfile(value):
			settings['fasta'] = value
		else:
			eprintWrap(f"FASTA file '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-h', '--help'):
		eprintWrap('A Python3 script for cleaning GenBank FASTA formated downloads.')
		eprintWrap(f"Output GenBank accessions only (optional; default = {settings['accessions']}): -a | --accessions")
		eprintWrap(fastaError)
		sys.exit(0)



### START/END
if not settings['fasta']:
	eprintWrap(fastaError)
	sys.exit(2)
else:
	eprintWrap('Started...')
	for key, value in settings.items():
		eprintWrap(f"{key} = {value}")



### CLEANING FUNCTIONS
def breakLines(sequence):
	output = ''
	for k, base in enumerate(sequence):
		if k > 0 and k % settings['columns'] == 0:
			output += '\n'
		output += base
	return output

def output(definition, sequence):
	if len(definition) and len(sequence):
		words = definition.split(' ')
		if settings['accessions']:
			print(words[0])
		else:
			print(f">{species(words[1:])}_{re.sub(DEFLINE, '', words[0])}")
		print(breakLines(re.sub(CHAR, '', sequence.upper())))

def species(words): ### assumes properly formed names (not always true)
	if len(words) < 3: ### Genus | Genus specific
		return '_'.join(words)
	elif testWords(words, 0, 'x'): ### x Genus ...
		return f"×_{species('_'.join(words[1:]))}"
	elif testWords(words, 1, 'x'): ### Genus x specific ...
		return f"{words[0]}_×_{specificEpithet(words[2:])}"
	elif len(words) >= 4 and testWords(words, 2, 'x'): ### Genus specific x specific ...
		return f"{words[0]}_{words[1]}_×_{specificEpithet(words[3:])}"
	else: ### Genus specificEpithet ...
		return f"{words[0]}_{specificEpithet(words[1:])}" 

def specificEpithet(words):
	if len(words) == 0:
		return ''
	elif len(words) >= 3 and re.search(INTRA, words[1]): ### specific intra intraspecific ...
		return f"{words[0]}_{words[1]}_{specificEpithet(words[2:])}"
	else: ### specificEpithet ...
		return words[0]

def testWords(words, index, value):
	try:
		return words[index] == value
	except IndexError:
		return False



### READ AND OUTPUT
definition = ''
sequence = ''
with open(settings['fasta'], mode = 'rt', encoding = 'utf8', errors = 'replace') as file:
	for line in file:
		text = ftfy.fix_text(line).strip()
		if re.search(DEFLINE, text):
			output(definition, sequence)
			definition = text
			sequence = ''
		else:
			sequence += text
output(definition, sequence)
