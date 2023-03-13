#!/usr/bin/env python3

### REQUIRED IMPORTS
import ftfy
import getopt
import re
import shutil
import sys
import textwrap



### CONSTANTS
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



### PRINT NULL
def nullPrint(*args, **kwargs):
	print(*args, end = '\0', **kwargs)



### USER SETTINGS
settings = {}
settings['gene'] = ''



### READ OPTIONS
geneError = 'Gene query string (required): -g query | --gene query'
try:
	arguments, values = getopt.getopt(sys.argv[1:], 'g:h', ['gene=', 'help'])
except getopt.error as err:
	eprintWrap(str(err))
	sys.exit(2)
for argument, value in arguments:
	if argument in ('-g', '--gene'):
		# gene = '^gene\t' + re.escape(value) + '$'
		# settings['gene'] = re.compile(gene)
		settings['gene'] = re.compile('^gene\t' + re.escape(value) + '$')
	elif argument in ('-h', '--help'):
		eprintWrap('A Python3 script for filtering EDirect e-utilities feature tables (ft) for a particular gene.')
		eprintWrap("The script is designed to be used in a search pipeline: esearch -db nuccore -query <query> | efetch -format ft | ftFilter.py -g <gene> | xargs -0 -I {} -P 1 bash -c 'ACCESSION=$(echo \"{}\" | awk \"{print \$1}\"); START=$(echo \"{}\" | awk \"{print \$2}\"); STOP=$(echo \"{}\" | awk \"{print \$3}\"); esearch -db nuccore -query \"$ACCESSION\" | efetch -format fasta -seq_start $START -seq_stop $STOP'")
		eprintWrap(geneError)
		sys.exit(0)



### START/END
if not settings['gene']:
	eprintWrap(geneError)
	sys.exit(2)
else:
	eprintWrap('Started...')
	for key, value in settings.items():
		eprintWrap(f"{key} = {value}")



### CLEANING FUNCTIONS
def output(definition, table):
	if len(definition) and len(table):
		definitionWords = definition.split('|')
		tableLines = table.split('\n')
		for k, line in enumerate(tableLines):
			if re.search(settings['gene'], line):
				start, stop, cds = tableLines[k-1].split('\t')
				nullPrint(f"{definitionWords[1]}[ACCESSION] {start} {stop}")
				break



### READ AND OUTPUT
definition = ''
table = ''
for line in sys.stdin:
	text = ftfy.fix_text(line).strip()
	if re.search(DEFLINE, text):
		output(definition, table)
		definition = text
		table = ''
	else:
		table += f"{text}\n"
output(definition, table)
