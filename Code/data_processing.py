import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

data=pd.read_csv("../data/sentences.csv",sep="\t",encoding="utf8",index_col=0,names=['lang','text'])

#filter by lenth between 20 to 200
len_filt=[True if 20<=l<=200 else False for l in len(data[text])]
data=data[len_filt]

#filter by language,we want to keep only 6 languages
lang=['deu','eng','fra','ita','por','spa']
data=data[data[lang].isin(lang)]

#trim data to keep only 5000 sentences per language
trimmed_data=pd.DataFrame(columns=['lang','text'])
for l in lang:
    trimmed_lang=data[data[lang]==l].sample(5000,random_state=100)
    trimmed_data.append(trimmed_lang)

shuffled_data=trimmed_data.sample(frac=1)

train=shuffled_data[0:21000]
valid=shuffled_data[21000:27000]
test=shuffled_data[27000:]


def get_features(corpus,n_feat=200):

    vectorizer=CountVectorizer(analyzer="char",ngram_range=(3,3),max_features=n_feat)
    X=vectorizer.fit_transform(corpus)

    feature_names=vectorizer.get_feature_names()

    return feature_names

#obtain trigrams for each languages
features={}
feature_set=set()

for l in lang:
    corpus=train[train['lang']==l]['text']
    trigrams=get_features(corpus)

    features[l]=trigrams
    feature_set.update(trigrams)

#create vocabulary list using feature set,looks like {"trigram1":0,"trigram2":1,...}
vocab=dict()
for i,f in enumerate(feature_set):
    vocab[f]=i

#create matrix of all the sentences using vocabulary

vectorizer=CountVectorizer(analyzer='char',ngram_range=(3,3),vocabulary=vocab)

corpus=train['text']
X=vectorizer.fit_transform(corpus) 

feature_names=vectorizer.get_feature_names()

train_feat=pd.DataFrame(data=X.toarray(),columns=feature_names)

#scale the feature matrix
train_min=train_feat.min()
train_max=train_feat.max()
train_feat=(train_feat-train_min)/(train_max-train_min)

#add target variable
train_feat['lang']=list(train['lang'])

#create feature matrix of validation data
corpus=valid['text']
X=vectorizer.fit_transform(corpus)

valid_feat=pd.DataFrame(data=X.toarray(),columns=['lang','text'])
valid_feat=(valid_feat-train_min)/(train_max-train_min)
valid_feat['lang']=list(valid['lang'])

#create feature matrix of test data
corpus=test['text']
X=vectorizer.fit_transform(corpus)

test_feat=pd.DataFrame(data=X.toarray(),columns=['lang','text'])
test_feat=(test_feat-train_min)/(train_max-train_min)
test_feat['lang']=list(test['lang'])

#one hot encoding of output labels
#fit encoder

encoder=LabelEncoder()
encoder.fit(['deu','eng','fra','ita','por','spa'])

def encode(y):
    """
    y: list of language labels
    returns a list of one hot encoding of language list

    """
    y_encoded=encoder.transform(y)
    y_dummy=np_utils.to_categorical(y_encoded)

    return y_dummy
    