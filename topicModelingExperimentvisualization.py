import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
%matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
	# Init output
	sent_topics_df = pd.DataFrame()

	# Get main topic in each document
	for i, row in enumerate(ldamodel[corpus]):
	row = sorted(row, key=lambda x: (x[1]), reverse=True)
	# Get the Dominant topic, Perc Contribution and Keywords for each document
	for j, (topic_num, prop_topic) in enumerate(row):
		if j == 0:  # => dominant topic
			wp = ldamodel.show_topic(topic_num)
			topic_keywords = ", ".join([word for word, prop in wp])
			sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
		else:
			break
	sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

	# Add original text to the end of the output
	contents = pd.Series(texts)
	sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
	return(sent_topics_df)


def sent_to_words(sentences):
	for sentence in sentences:
		yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
	return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    	return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    	return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    	"""https://spacy.io/api/annotation"""
    	texts_out = []
    	for sent in texts:
        	doc = nlp(" ".join(sent)) 
        	texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    	return texts_out

bfinputdir = '/home/mike/QA_Project/lda_Data/Bug_Fix_sentiment_Data/'
biinputdir = '/home/mike/QA_Project/lda_Data/Bug_intro_sentiment_Data/'
bfout = '/home/mike/QA_Project/lda_Data/Bug_Fix_topic_output/'
biout = '/home/mike/QA_Project/lda_Data/Bug_intro_topic_output/'
bf_list= []
bi_list= []
total_list= []
for file in os.listdir(bfinputdir):
	file
	with open (bfinputdir+file, "r", encoding="utf8") as f:
		list1= []
		name= file.split('.')[0] 
		new_file = open(bfout+name+'_Lda1.txt','w')
		for line in f:	
			line=line.rstrip()			
			list1.append(line)
			bf_list.append(line)
			total_list.append(line)
		
		# Remove Emails
		list1 = [re.sub('\S*@\S*\s?', '', sent) for sent in list1]

		# Remove new line characters
		list1 = [re.sub('\s+', ' ', sent) for sent in list1]

		# Remove distracting single quotes
		list1 = [re.sub("\'", "", sent) for sent in list1]

		pprint(list1[:1])
		data_words = list(sent_to_words(list1))

		print(data_words[:1])
		# Build the bigram and trigram models
		bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
		trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

		# Faster way to get a sentence clubbed as a trigram/bigram
		bigram_mod = gensim.models.phrases.Phraser(bigram)
		trigram_mod = gensim.models.phrases.Phraser(trigram)

		# See trigram example
		print(trigram_mod[bigram_mod[data_words[0]]])
		# Remove Stop Words
		data_words_nostops = remove_stopwords(data_words)

		# Form Bigrams
		data_words_bigrams = make_bigrams(data_words_nostops)

		# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
		# python3 -m spacy download en
		nlp = spacy.load('en', disable=['parser', 'ner'])

		# Do lemmatization keeping only noun, adj, vb, adv
		data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

		print(data_lemmatized[:1])
		# Create Dictionary
		id2word = corpora.Dictionary(data_lemmatized)

		# Create Corpus
		texts = data_lemmatized

		# Term Document Frequency
		corpus = [id2word.doc2bow(text) for text in texts]

		# View
		print(corpus[:1])
		# Build LDA model
		lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
				                           id2word=id2word,
				                           num_topics=20, 
				                           random_state=100,
				                           update_every=1,
				                           chunksize=100,
				                           passes=10,
				                           alpha='auto',
				                           per_word_topics=True)
		# Print the Keyword in the 10 topics
		pprint(lda_model.print_topics())
		doc_lda = lda_model[corpus]
		# Compute Perplexity
		print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

		# Compute Coherence Score
		coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
		coherence_lda = coherence_model_lda.get_coherence()
		print('\nCoherence Score: ', coherence_lda)
		# Visualize the topics
		pyLDAvis.enable_notebook()
		vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
		vis
		df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)

		# Format
		df_dominant_topic = df_topic_sents_keywords.reset_index()
		df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

		# Show
		df_dominant_topic.head(10)
		# Number of Documents for Each Topic
		topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

		# Percentage of Documents for Each Topic
		topic_contribution = round(topic_counts/topic_counts.sum(), 4)

		# Topic Number and Keywords
		topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

		# Concatenate Column wise
		df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

		# Change Column names
		df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

		# Show
		df_dominant_topics

