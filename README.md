# Deep Neural Network Language Identification

Language identification can be an important step in a Natural Language Processing (NLP) problem. It involves trying to predict the natural language of a piece of text. It is important to know the language of text before other actions (i.e. translation/ sentiment analysis) can be taken. For instance, if you go to google translate the box you type in says ‘Detect Language’. This is because Google is first trying to identify the language of your sentence before it can be translated.

![animation](/images/animation.gif)

There are several different approaches to language identification and, in this article, we’ll explore one in detail. That is using a Neural Network and character n-grams as features. In the end, we show that an accuracy of over 98% can be achieved with this approach.Firstly, we’ll discuss the dataset that we’ll use to train our Neural Network.



## Dataset

The dataset is provided by [Tatoeba](https://downloads.tatoeba.org/exports/). The full dataset consists of 6,872,356 sentences in 328 unique languages. To simplify our problem we will consider:
6 Latin languages: English, German, Spanish, French, Portuguese and Italian.
Sentences between 20 and 200 characters long.

We can see an example of a sentence from each language in Table 1. Our objective is to create a model that can predict the Target Variable using the Text provided.
| Language  | Target Variable  |      Text      |
| ----------| -------- | ------------------------------------------------- |
| German    | due      | Ich war um seine Gesundheit besorgt.              |
| English   | eng      | He doffed his hat when he saw me.                 |
| French    | fra      | Il a ce qu'il faut pour réussir dans le monde  des affaires.  |
| Italian   | ita      | Non lo augurerei a nessuno.                       |
| Portuguese| por      | Cante-me uma canção de ninar.                     |
| Spanish   | spa      | El surgimiento del exoesqueleto en los artrópodos  fue un acontecimiento evolutivo muy importante para esos animales |

**Data Processing**

We load the dataset and do some initial processing in the code below. We first filter the dataset to get sentences of the desired length and language. We randomly select 50,000 sentences from each of these languages so that we have 300,000 rows in total. These sentences are then split into a training (70%), validation (20%) and test (10%) set.

**Feature Engineering**

For our language identification problem, we will be using character 3-grams/trigrams (i.e. sets of 3 consecutive characters). In Figure 2, we see an example of how sentences can be vectorised using trigrams. Firstly, we get all the trigrams from the sentences. To reduce the feature space, we take a subset of these trigrams. We use this subset to vectorise the sentences. The vector for the first sentence is [2,0,1,0,0] as the trigram ‘is_’ occurs twice and ‘his’ occurs once in the sentence.

1.Using the training set, we select the 200 most common trigrams from each language
2.Create a list of unique trigrams from these trigrams. The languages share a few common trigrams and so we end up with a 663 unique trigrams
3.Create a feature matrix, by counting the number of times each trigram occurs in each sentence

We can see an example of such a feature matrix in Table 2. The top row gives each of the 663 trigrams. Then each of the numbered rows gives one of the sentences in our dataset. The numbers within the matrix give the number of times that trigram occurs within the sentence. For example, “j’a” occurs once in sentence 2.

|ach  |abe  |  j'a |   ux | ion | en, | est | aun | ...|  
| ---- |----| -----|------|---- |-----|-----|-----|----|
|0 | 0 | 0 | 0 | 0 | 0 | 0| 0 | ... |
|0 | 0 | 0 | 0 | 0 | 0 | 0| 0 | ... |
|0 | 0 | 1 | 0 | 0 | 0 | 0| 0 | ... |
|0 | 0 | 0 | 0 | 0 | 0 | 0| 0 | ... |
|0 | 0 | 0 | 0 | 0 | 0 | 0| 2 | ... |
|... | ... | .. | ... | ... | ... | ...| ... | ... |

**Exploring Trigrams**

We now have the datasets in a form ready to be used to train our Neural Network. Before we do that, it would be useful to explore the dataset and build up a bit of an intuition around how well these features will do at predicting the languages. Figure 2 gives the number of trigrams each language has in common with the others. For example, English and German have 55 of their most common trigrams in common.
With 127, we see that Spanish and Portuguese have the most trigrams in common. This makes sense as, among all the languages, these two are the most lexically similar. What this means is that, using these features, our model may find it difficult to distinguish Spanish from Portuguese and visa versa. Similarly, Portuguese and German have the least trigrams in common and we could expect our model to be better at distinguishing these languages.

![Trigrams](/images/trigram.png)

**Modelling**

We use the keras package to train our DNN. A softmax activation function is used in the model’s output layer. This means we have to transform our list of target variables into a list of one-hot encodings. This is done using the encode function below. This function takes a list of target variables and returns a list of one-hot encoded vectors. For example, [eng,por,por, fra,…] would become [[0,1,0,0,0,0],[0,0,0,0,1,0],[0,0,0,0,1,0],[0,0,1,0,0,0],…].

We use the keras package to train our DNN. A softmax activation function is used in the model’s output layer. This means we have to transform our list of target variables into a list of one-hot encodings. This is done using the encode function below. This function takes a list of target variables and returns a list of one-hot encoded vectors. For example, [eng,por,por, fra,…] would become [[0,1,0,0,0,0],[0,0,0,0,1,0],[0,0,0,0,1,0],[0,0,1,0,0,0],…].

```python
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

#Fit encoder
encoder = LabelEncoder()
encoder.fit(['deu', 'eng', 'fra', 'ita', 'por', 'spa'])

def encode(y):
    """
    Returns a list of one hot encodings
    Params
    ---------
        y: list of language labels
    """
    
    y_encoded = encoder.transform(y)
    y_dummy = np_utils.to_categorical(y_encoded)
    
    return y_dummy
}
```
The final model has 3 hidden layers with 500, 500 and 250 nodes respectfully. The output layer has 6 nodes, one for each language. The hidden layers all have ReLU activation functions and, as mentioned, the output layer has a softmax activation function. We train this model using 4 epochs and a batch size of 100. Using our training set and one-hot encoded target variable list, we train this DDN in the code below. In the end, we achieve a training accuracy of 99.70%.


**Model evaluation**

During the model training process, the model can become biased towards the training set as well as the validation set. So it is best to determine the model accuracy on an unseen test set. The final accuracy on the test set was 98.26%. This is lower than the training accuracy of 99.70% suggesting that some overfitting to the training set has occurred.
We can get a better idea of how well the model does for each language by looking at the confusion matrix in Figure . The red diagonal gives the number of correct predictions for each language. The off-diagonal numbers give the number of times a language was incorrectly predicted as another. For example, German is incorrectly predicted as English 10 times. We see that the model most often confuses either Portuguese for Spanish (124 times) or Spanish for Portuguese (61 times). This follows from what we saw when exploring our features.

![Confusion matrix](/images/confusion.png)

In the end, the test accuracy of 98.26% leaves room for improvement. In terms of feature selection, we have kept things simple and have just selected the 200 most common trigrams for each language. A more complicated approach could help us differentiate the languages that are more similar. For example, we could select trigrams that are common in Spanish but not so common in Portuguese and visa versa. We could also experiment with different models. Hopefully, this is a good starting point for your language identification experiments.


