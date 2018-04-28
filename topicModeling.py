import pandas as pd
import os.path as path
import os
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()



def parse_comment_file(filepath):
	df= pd.read_csv(filepath, header=None, delim_whitespace=True ,skiprows=1)
	with open(filepath) as f:
		lines = f.readlines()
		header = lines[0].strip().replace("#", "").strip()
		columns = header.split
		f.close
	#df.columns = ["sha","comment", "positive", "negative"]
	return df

def display_topics(model, feature_names, no_top_words):
	for topic_idx, topic in enumerate(model.components_):
        	print ("Topic %d:" % (topic_idx))
        	print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

def clean(doc):
	stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
	punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
	normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
	return normalized

bfinputdir = '/home/mike/QA_Project/lda_Data/Bug_Fix_sentiment_Data/'
biinputdir = '/home/mike/QA_Project/lda_Data/BI_SentimentsDetection/'
bfout = '/home/mike/QA_Project/lda_Data/Bug_Fix_sentiment_output/'
biout = '/home/mike/QA_Project/lda_Data/Bug_intro_output/'

with open (bfinputdir+'BF_atmosphere.txt') as f:
	list= []
	#name= file.split('.')[0] 		
	for line in f:	
		line=line.rstrip()			
		list.append(line)


doc_clean = [clean(doc).split() for doc in list]   
# Importing Gensim
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.S
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)
print(ldamodel.print_topics(num_topics=3, num_words=3))


no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(list)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(list)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20
# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
lda2 = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

no_top_words = 10
display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda2, tf_feature_names, no_top_words)
