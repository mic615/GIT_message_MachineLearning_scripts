import pandas as pd
import os.path as path
import os
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
# Importing Gensim
import gensim
from gensim import corpora
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def format_topics_sentences(ldamodel, corpus, texts):
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


def clean(doc):
	stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
	punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
	normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
	return normalized

bfinputdir = '/home/mike/QA_Project/lda_Data/Bug_Fix_sentiment_Data/'
biinputdir = '/home/mike/QA_Project/lda_Data/Bug_intro_sentiment_Data/'
bfout = '/home/mike/QA_Project/lda_Data/Bug_Fix_20topic_output/'
biout = '/home/mike/QA_Project/lda_Data/Bug_intro_20topic_output/'
bf_list= []
bi_list= []
total_list= []
for file in os.listdir(bfinputdir):
	file
	with open (bfinputdir+file, "r", encoding="utf8") as f:
		list= []
		name= file.split('.')[0] 
		new_file = open(bfout+name+'_Lda1.txt','w')		
		for line in f:	
			line=line.rstrip()			
			list.append(line)
			bf_list.append(line)
			total_list.append(line)
		doc_clean = [clean(doc).split() for doc in list]
		# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
		dictionary = corpora.Dictionary(doc_clean)	
		# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
		doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
		# Creating the object for LDA model using gensim library
		Lda = gensim.models.ldamodel.LdaModel
		# Running and Trainign LDA model on the document term matrix.S
		ldamodel = Lda(doc_term_matrix, num_topics=20, id2word = dictionary, passes=50)
		topics=ldamodel.print_topics(num_topics=20, num_words=10)
		for topic in topics:
			new_file.write(str(topic))
			new_file.write('\n')
		print(ldamodel.print_topics(num_topics=20, num_words=10))
		# Compute Perplexity
		#print('\nPerplexity: ', ldamodel.log_perplexity(dictionary))  # a measure of how good the model is. lower the better.
		#new_file.write('\nPerplexity: ', ldamodel.log_perplexity(dictionary))
		# Compute Coherence Score
		#coherence_model_lda = CoherenceModel(model=ldamodel, texts=list, dictionary=id2word, coherence='c_v')
		#coherence_lda = coherence_model_lda.get_coherence()
		#print('\nCoherence Score: ', coherence_lda)
		#new_file.write('\nCoherence Score: ', coherence_lda)
		new_file.close()
		#df_topic_sents_keywords = format_topics_sentences(ldamodel, dictionary, list)
		# Init output
		sent_topics_df = pd.DataFrame()
		# Get main topic in each document
		for i, row in enumerate(ldamodel[doc_term_matrix]):
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
		contents = pd.Series(list)
		sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
		sent_topics_df
		# Format
		df_dominant_topic = sent_topics_df.reset_index()
		df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
		df_dominant_topic
		# Show
		df_dominant_topic.head(10)
		df_dominant_topic.to_csv(bfout+name+'Dominant_Topics_Lda1.txt', sep=',',header=False,index=False)
		
## bug intro comments
for file in os.listdir(biinputdir):
	file
	with open (biinputdir+file, "r", encoding="utf8") as f:
		list= []
		name= file.split('.')[0] 
		new_file = open(biout+name+'_Lda1.txt','w')		
		for line in f:	
			line=line.rstrip()			
			list.append(line)
			bi_list.append(line)
			total_list.append(line)
		doc_clean = [clean(doc).split() for doc in list]
		# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
		dictionary = corpora.Dictionary(doc_clean)	
		# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
		doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
		# Creating the object for LDA model using gensim library
		Lda = gensim.models.ldamodel.LdaModel
		# Running and Trainign LDA model on the document term matrix.S
		ldamodel = Lda(doc_term_matrix, num_topics=20, id2word = dictionary, passes=50)
		topics=ldamodel.print_topics(num_topics=20, num_words=10)
		for topic in topics:
			new_file.write(str(topic))
			new_file.write('\n')
		print(ldamodel.print_topics(num_topics=20, num_words=10))
		# Compute Perplexity
		#print('\nPerplexity: ', ldamodel.log_perplexity(dictionary))  # a measure of how good the model is. lower the better.
		#new_file.write('\nPerplexity: ', ldamodel.log_perplexity(dictionary))
		# Compute Coherence Score
		#coherence_model_lda = CoherenceModel(model=ldamodel, texts=list, dictionary=id2word, coherence='c_v')
		#coherence_lda = coherence_model_lda.get_coherence()
		#print('\nCoherence Score: ', coherence_lda)
		#new_file.write('\nCoherence Score: ', coherence_lda)
		new_file.close()
		#df_topic_sents_keywords = format_topics_sentences(ldamodel, dictionary, list)
		# Init output
		sent_topics_df = pd.DataFrame()
		# Get main topic in each document
		for i, row in enumerate(ldamodel[doc_term_matrix]):
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
		contents = pd.Series(list)
		sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
		sent_topics_df
		# Format
		df_dominant_topic = sent_topics_df.reset_index()
		df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
		df_dominant_topic
		# Show
		df_dominant_topic.head(10)
		df_dominant_topic.to_csv(biout+name+'Dominant_Topics_Lda1.txt', sep=',',header=False,index=False)
		

#Bf
new_file = open(bfout+'total_BF_Lda1.txt','w')
doc_clean = [clean(doc).split() for doc in bf_list]
# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(doc_clean)	
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel
# Running and Trainign LDA model on the document term matrix.S
ldamodel = Lda(doc_term_matrix, num_topics=20, id2word = dictionary, passes=50)
topics=ldamodel.print_topics(num_topics=20, num_words=10)
for topic in topics:
	new_file.write(str(topic))
	new_file.write('\n')
print(ldamodel.print_topics(num_topics=20, num_words=10))
# Compute Perplexity
#print('\nPerplexity: ', ldamodel.log_perplexity(dictionary))  # a measure of how good the model is. lower the better.
#new_file.write('\nPerplexity: ', ldamodel.log_perplexity(dictionary))
# Compute Coherence Score
#coherence_model_lda = CoherenceModel(model=ldamodel, texts=list, dictionary=id2word, coherence='c_v')
#coherence_lda = coherence_model_lda.get_coherence()
#print('\nCoherence Score: ', coherence_lda)
#new_file.write('\nCoherence Score: ', coherence_lda)
new_file.close()
#df_topic_sents_keywords = format_topics_sentences(ldamodel, dictionary, list)
# Init output
sent_topics_df = pd.DataFrame()
# Get main topic in each document
for i, row in enumerate(ldamodel[doc_term_matrix]):
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
contents = pd.Series(bf_list)
sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
sent_topics_df
# Format
df_dominant_topic = sent_topics_df.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic
# Show
df_dominant_topic.head(10)
df_dominant_topic.to_csv(bfout+'total_BF_Dominant_Topics_Lda1.txt', sep=',',header=False,index=False)
		

#Bi
new_file = open(biout+'total_BI_Lda1.txt','w')
doc_clean = [clean(doc).split() for doc in bi_list]
# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(doc_clean)	
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel
# Running and Trainign LDA model on the document term matrix.S
ldamodel = Lda(doc_term_matrix, num_topics=20, id2word = dictionary, passes=50)
topics=ldamodel.print_topics(num_topics=20, num_words=10)
for topic in topics:
	new_file.write(str(topic))
	new_file.write('\n')
print(ldamodel.print_topics(num_topics=20, num_words=10))
# Compute Perplexity
#print('\nPerplexity: ', ldamodel.log_perplexity(dictionary))  # a measure of how good the model is. lower the better.
#new_file.write('\nPerplexity: ', ldamodel.log_perplexity(dictionary))
# Compute Coherence Score
#coherence_model_lda = CoherenceModel(model=ldamodel, texts=list, dictionary=id2word, coherence='c_v')
#coherence_lda = coherence_model_lda.get_coherence()
#print('\nCoherence Score: ', coherence_lda)
#new_file.write('\nCoherence Score: ', coherence_lda)
new_file.close()
#df_topic_sents_keywords = format_topics_sentences(ldamodel, dictionary, list)
# Init output
sent_topics_df = pd.DataFrame()
# Get main topic in each document
for i, row in enumerate(ldamodel[doc_term_matrix]):
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
contents = pd.Series(bi_list)
sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
sent_topics_df
# Format
df_dominant_topic = sent_topics_df.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic
# Show
df_dominant_topic.head(10)
df_dominant_topic.to_csv(biout+'total_BI_Dominant_Topics_Lda1.txt', sep=',',header=False,index=False)

#total
new_file = open(biout+'total_dataset_Lda1.txt','w')
doc_clean = [clean(doc).split() for doc in total_list]
# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(doc_clean)	
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel
# Running and Trainign LDA model on the document term matrix.S
ldamodel = Lda(doc_term_matrix, num_topics=20, id2word = dictionary, passes=50)
topics=ldamodel.print_topics(num_topics=20, num_words=10)
for topic in topics:
	new_file.write(str(topic))
	new_file.write('\n')
print(ldamodel.print_topics(num_topics=20, num_words=10))
# Compute Perplexity
#print('\nPerplexity: ', ldamodel.log_perplexity(dictionary))  # a measure of how good the model is. lower the better.
#new_file.write('\nPerplexity: ', ldamodel.log_perplexity(dictionary))
# Compute Coherence Score
#coherence_model_lda = CoherenceModel(model=ldamodel, texts=list, dictionary=id2word, coherence='c_v')
#coherence_lda = coherence_model_lda.get_coherence()
#print('\nCoherence Score: ', coherence_lda)
#new_file.write('\nCoherence Score: ', coherence_lda)
new_file.close()
#df_topic_sents_keywords = format_topics_sentences(ldamodel, dictionary, list)
# Init output
sent_topics_df = pd.DataFrame()
# Get main topic in each document
for i, row in enumerate(ldamodel[doc_term_matrix]):
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
contents = pd.Series(total_list)
sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
sent_topics_df
# Format
df_dominant_topic = sent_topics_df.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic
# Show
df_dominant_topic.head(10)
df_dominant_topic.to_csv(biout+'total_dataset_Dominant_Topics_Lda1.txt', sep=',',header=False,index=False)
