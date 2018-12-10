
# coding: utf-8

# In[23]:


import numpy as np
import re
import pandas as pd 
import matplotlib as mp
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


test = pd.read_csv("C:/Users/Sdeol/Desktop/DS/test_tweets_anuFYb8.csv")
test.head()


# In[25]:


train = pd.read_csv("C:/Users/Sdeol/Desktop/DS/train_E6oV3lV.csv")
train.head()


# In[26]:


Df = train.append(test, ignore_index = True)


# In[27]:


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt   


# In[28]:


Df['clean_tweet'] = np.vectorize(remove_pattern)(Df['tweet'], "@[\w]*")


# In[29]:


Df['clean_tweet'] = Df['clean_tweet'].str.replace("[^a-zA-Z#]", " ")


# In[30]:


Df.head()


# In[31]:


Df['clean_tweet'] = Df['clean_tweet'].apply(lambda x:' '.join(w for w in x.split() if len(w)>3))


# In[32]:


Df.head()


# In[33]:


#Split words-Tokenize
split_tweet = Df['clean_tweet'].apply(lambda x: x.split())
split_tweet.head()


# In[34]:


from nltk.stem.porter import *
stemmer = PorterStemmer()

split_tweet = split_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 
# stemming
split_tweet.head()


# In[35]:


for i in range(len(split_tweet)):
    split_tweet[i] = ' '.join(split_tweet[i])

Df['clean_tweet'] = split_tweet


# In[36]:


print (Df["clean_tweet"].head())


# In[37]:


visual_words = ' '.join([text for text in Df['clean_tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(visual_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[38]:


love_words =' '.join([text for text in Df['clean_tweet'][Df['label'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(love_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[39]:


hate_words =' '.join([text for text in Df['clean_tweet'][Df['label'] == 1]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(hate_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[40]:


# function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags


# In[41]:


# extracting hashtags from non racist/sexist tweets

HT_regular = hashtag_extract(Df['clean_tweet'][Df['label'] == 0])

# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(Df['clean_tweet'][Df['label'] == 1])

# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])


# In[43]:


from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(Df['clean_tweet'])
bow


# In[45]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(Df['clean_tweet'])
tfidf


# In[46]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = bow[:31962,:]
test_bow = bow[31962:,:]

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model

prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int) # calculating f1 score

