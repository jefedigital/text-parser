# import modules
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.probability import FreqDist
from nltk import bigrams, trigrams
from nltk.collocations import *
import glob

# import stopwords corpus
stopwords = nltk.corpus.stopwords.words('english')

# define data sources
dir = 'data/joblistings/director_analytics/*.txt'
file = 'data/joblistings/sample_job.txt'

# init
tokens = []
raw=''

# start file open routine
print('Loading files:\n')

## if single file, uncomment these lines
## f = open(file)
## raw = f.read()

## if multiple files, uncomment these lines
files = glob.glob(dir)
for name in files:
    print(name)
    with open(name) as f:
        raw += ('\n' + f.read())

# end file open routine

raw_tokens = WhitespaceTokenizer().tokenize(raw)

# filter out stopwords
for t in raw_tokens:
    if t.lower() not in stopwords:
        tokens.append(t.lower())

# build nltk units
raw_text = nltk.Text(raw_tokens)
text = nltk.Text(tokens)
vocab = sorted(set(text))
dist = FreqDist(text)

# stats
print('Total words (unfiltered):', len(raw_tokens))
print('Total words (filtered):', len(tokens))

# Top keywords
common_25 = dist.most_common(25)

print('Top keywords & count:\n')
for x in common_25:
    print(': '.join(map(str,x)))

# Find collocations

## bigrams
bigram_measures = nltk.collocations.BigramAssocMeasures()
bi_finder = BigramCollocationFinder.from_words(text, 2) # search window
bi_finder.apply_freq_filter(3) # number of recurrences
bigrams = bi_finder.nbest(bigram_measures.likelihood_ratio, 1000)

print('Top Bigrams:\n',)
for x in bigrams:
    print(' '.join(x))

## trigrams
trigram_measures = nltk.collocations.TrigramAssocMeasures()
tri_finder = TrigramCollocationFinder.from_words(text, 3)
tri_finder.apply_freq_filter(2)
trigrams = tri_finder.nbest(trigram_measures.likelihood_ratio, 1000)

print('Top Trigrams:\n',)
for x in trigrams:
    print(' '.join(x))
