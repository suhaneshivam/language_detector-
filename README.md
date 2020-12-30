# Deep Neural Network Language Identification
## Dataset

The dataset is provided by Tatoeba. The full dataset consists of 6,872,356 sentences in 328 unique languages. To simplify our problem we will consider:
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




