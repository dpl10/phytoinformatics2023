#!/usr/bin/env python3

### SAFE IMPORTS
import getopt
import os
import re
import shutil
import sys
import textwrap



### CONSTANTS
VERSION = re.compile('\.[0-9]+$')
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
settings['diamondFile'] = ''
settings['goFile'] = ''



### READ OPTIONS
diamondFileError = "Input DIAMOND .tsv file (required; produced with '-f 6 qseqid sseqid'; query must be the first column, subject must be the second column): -d file.tsv | --diamond=file.tsv"
goFileError = 'Input GO annotations .tsv file (required; subject must be in the first column, GO annotations in the subsequent columns): -g file.tsv | --go=file.tsv'
try:
	arguments, values = getopt.getopt(sys.argv[1:], 'd:g:h', ['diamond=', 'go=', 'help'])
except getopt.error as err:
	eprintWrap(str(err))
	sys.exit(2)
for argument, value in arguments:
	if argument in ('-d', '--diamond'):
		if os.path.isfile(value):
			settings['diamondFile'] = value
		else:
			eprintWrap(f"Input DIAMOND file does not exist '{value}'!")
			sys.exit(2)
	if argument in ('-g', '--go'):
		if os.path.isfile(value):
			settings['goFile'] = value
		else:
			eprintWrap(f"Input GO file does not exist '{value}'!")
			sys.exit(2)
	elif argument in ('-h', '--help'):
		eprintWrap('')
		eprintWrap('A Python3 script to add GO annotations to DIAMOND (or BLAST) output.')
		eprintWrap(diamondFileError)
		eprintWrap(goFileError)
		eprintWrap('')
		sys.exit(0)



### START/END
if not settings['diamondFile']:
	eprintWrap(diamondFileError)
	sys.exit(2)
elif not settings['goFile']:
	eprintWrap(goFileError)
	sys.exit(2)
else:
	eprintWrap('started...')
	for key, value in settings.items():
		eprintWrap(f"{key} = {value}")
	eprintWrap('')



### GET GO
go = {} ### subject => [GO annotations, ...]
with open(settings['goFile'], mode = 'rt', encoding = 'utf8', errors = 'replace') as file:
	for line in file:
		columns = line.strip().split('\t')
		if columns[0] not in go:
			go[columns[0]] = []
		go[columns[0]].append('\t'.join(columns[1:]))



### JOIN AND OUTPUT
with open(settings['diamondFile'], mode = 'rt', encoding = 'utf8', errors = 'replace') as file:
	for line in file:
		columns = line.strip().split('\t')
		columns[1] = re.sub(VERSION, '', columns[1])
		if columns[1] in go:
			print(f"{columns[0]}\t" + f"\n{columns[0]}\t".join(go[columns[1]]))
