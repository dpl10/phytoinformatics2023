#!/usr/bin/env python3

### IMPORTS
import Bio.Blast.Applications
import ftfy
import getopt
import multiprocessing
import os
import re
import textwrap
import sys
import xxhash

### GLOBAL CONSTANTS
BLASTFILE = 'temporary-do-not-touch'
COLUMNS = 80 ### .fasta line width
DEFLINE = re.compile('^>')
DNA = re.compile('[^ABCDGHKMNRSTVWY]')
EVALUE = 0.01
INTRA = re.compile('(convar\.|f\.|forma|grex|lusus|microgene|modif\.|monstr\.|mut\.|' \
	'nothosubsp\.|nothovar\.|proles|provar\.|spp\.|spp|stirps|subf\.|sublusus|subproles|' \
	'subso|subsp\.|subvar\.|var\.|var)' \
)
PIPE = re.compile('\|')
QUERYFILE = 'temporary-query-do-not-touch'
TARGETS = 1000
WRAP = int(os.popen('stty size', 'r').read().split()[1])
XML = re.compile('<|>')

### GLOBAL USER SETTINGS AND DEFAULTS
settings = {}
settings['cores'] = multiprocessing.cpu_count()
settings['database'] = ''
settings['seed'] = ''

### FUNCTIONS
def breakLines(sequence):
	output = ''
	for position, base in enumerate(sequence):
		if position > 0 and position % COLUMNS == 0:
			output += '\n'
		output += base
	return output

def eprint(*arguments, **keywordArguments):
	print(*arguments, file = sys.stderr, **keywordArguments)

def seedStore(seed, sequence):
	if len(sequence) > 0:
		cleanSequence = re.sub(DNA, '', sequence.upper())
		seed[xxhash.xxh128_hexdigest(cleanSequence)] = cleanSequence

def species(name): ### assumes properly formed names (not always true)
	words = name.split(' ')
	if len(words) < 3: ### Genus | Genus specific
		return name
	elif testWords(words, 0, 'x'): ### x Genus ...
		return f"× {species(' '.join(words[1:]))}"
	elif testWords(words, 1, 'x'): ### Genus x specific ...
		return f'{words[0]} × {specificEpithet(words[2:])}'
	elif len(words) >= 4 and testWords(words, 2, 'x'): ### Genus specific x specific ...
		return f'{words[0]} {words[1]} × {specificEpithet(words[3:])}'
	else: ### Genus specificEpithet ...
		return f'{words[0]} {specificEpithet(words[1:])}' 

def specificEpithet(words):
	if len(words) == 0:
		return ''
	elif len(words) >= 3 and re.search(INTRA, words[1]): ### specific intra intraspecific ...
		return f'{words[0]} {words[1]} {specificEpithet(words[2:])}'
	else: ### specificEpithet ...
		return words[0]

def testWords(words, index, value):
	try:
		return words[index] == value
	except IndexError:
		return False

def wrap(string, columns = WRAP):
	return '\n'.join(textwrap.wrap(string, columns))

### READ OPTIONS
databaseError = wrap('Database to be searched (required): -d name | --database=name')
seedError = wrap('Search seed (required): -s file.fasta | --seed=file.fasta')
try:
	arguments, values = getopt.getopt(
		sys.argv[1:], 
		'c:d:hs:', 
		['cores=', 'database=', 'help', 'seed=']
	)
except getopt.error as error:
	eprint(str(error))
	sys.exit(2)
for argument, value in arguments:
	if argument in ('-c', '--cores') and int(value) > 0:
		settings['cores'] = int(value)
	elif argument in ('-d', '--database'):
		extensions = ['ndb', 'nhd', 'nhi', 'nhr', 'nin', 'nos', 'not', 'nsq', 'ntf']
		allFiles = True
		for extension in extensions:
			if os.path.isfile(f'{value}.{extension}') == False:
				eprint(wrap(f"Database file '{value}.{extension}' is missing!"))
				allFiles = False
				break
		if allFiles:
			settings['database'] = value
	elif argument in ('-h', '--help'):
		eprint('')
		eprint(wrap('A Python3 script for exhaustively BLASTing a seed nucleotide sequence.'))
		eprint(wrap(f"Cores for BLAST+ (optional; default = {settings['cores']}):" \
			' -c int | --cores=int'
		))
		eprint(databaseError)
		eprint(seedError)
		eprint('')
		sys.exit(0)
	elif argument in ('-s', '--seed'):
		if os.path.isfile(value):
			settings['seed'] = value
		else:
			eprint(wrap(f"Seed file '{value}' does not exist!"))
			sys.exit(2)

### START OR END
if not settings['database']:
	eprint(databaseError)
	sys.exit(2)
elif not settings['seed']:
	eprint(seedError)
	sys.exit(2)
else:
	eprint('started...')
	for key, value in settings.items():
		eprint(f'{key} = {value}')

### LOAD INITIAL SEED
seed = {}
sequence = ''
with open(settings['seed'], mode = 'rt', encoding = 'utf8', errors = 'replace') as file:
	for line in file:
		text = ftfy.fix_text(line).rstrip()
		if re.search(DEFLINE, text):
			seedStore(seed, sequence)
			sequence = ''
		else:
			sequence += text
seedStore(seed, sequence)

### QUERY LOOP
data = {}
while len(seed) > 0:
	output = open(QUERYFILE, 'w')
	for hash, sequence in seed.items():
		output.write(f'>{hash}\n') 
		output.write(breakLines(sequence) + '\n') 
	output.close()
	eprint(f'{len(seed)} sequences for BLAST search')

	os.system(str(Bio.Blast.Applications.NcbiblastnCommandline(
		db = settings['database'], 
		evalue = EVALUE, 
		max_target_seqs = TARGETS, 
		num_threads = settings['cores'],
		out = BLASTFILE, 
		outfmt = 5, 
		query = QUERYFILE
	)))
	os.unlink(QUERYFILE)
	eprint('BLAST search complete')

	accession = ''
	seed = {}
	sequence = ''
	taxon = ''
	with open(BLASTFILE, mode = 'rt', encoding = 'utf8', errors = 'replace') as file:
		for line in file:
			tokens = re.split(XML, ftfy.fix_text(line).strip())
			if testWords(tokens, 1, 'Hit_num'):
				accession = ''
				sequence = ''
				taxon = ''
			elif testWords(tokens, 1, 'Hit_def'):
				taxon = species(tokens[2])
			elif testWords(tokens, 1, 'Hit_id'):
				if re.search(PIPE, tokens[2]):
					accession = tokens[2].split('|')[1]
				else:
					accession = tokens[2]
			elif testWords(tokens, 1, 'Hsp_hseq'):
				sequence = re.sub(DNA, '', tokens[2].upper()) 
				key = f'>{accession} {taxon}'
				if not key in data or (key in data and len(data[key]) < len(sequence)):
					data[key] = sequence
					seedStore(seed, sequence)
	os.unlink(BLASTFILE)

### OUTPUT
for accessionTaxon, sequence in data.items():
	print(accessionTaxon)
	print(breakLines(sequence))

sys.exit(0)
