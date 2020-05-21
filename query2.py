import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Constants
TRAIN = 0.75

# Load the dataset
data = pd.read_csv("onion-or-not.csv")
# Add all the required columns
# We initiate them as strings as dictionary
# initialization fails on Pandas DataFrame
data["words"] = ""
data["wc"] = ""
data["tf"] = ""
data["idf"] = ""
data["tfidf"] = ""

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
sentences_per_word = {}
for index, row in data.iterrows():
	# Tokenize sentences to word vectors
	words = word_tokenize(row["text"])
	# Stem the words
	words = [ps.stem(w) for w in words]
	# Remove stopwords
	words = [w for w in words if not w in stop_words]
	# Calculate the term frequency
	word_count = {}
	for w in words:
		if w in word_count:
			word_count[w] += 1
		else:
			word_count[w] = 1
	row["words"] = words
	row["wc"] = word_count
	row["tf"] = {k: v / len(words) for k, v in word_count.items()}
	for word, count in word_count.items():
		if word in sentences_per_word:
			sentences_per_word[word] += count
		else:
			sentences_per_word[word] = count
	break

sentences = len(data)

for index, row in data.iterrows():
	idf_table = {}
	for w in row["words"]:
		idf_table[w] = math.log(sentences / sentences_per_word[w])
	row["idf"] = idf_table


print(len(sentences_per_word))
print("done")