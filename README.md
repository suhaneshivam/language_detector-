# Deep Neural Network Language Identification
## Dataset
The dataset is provided by Tatoeba. The full dataset consists of 6,872,356 sentences in 328 unique languages. To simplify our problem we will consider:
6 Latin languages: English, German, Spanish, French, Portuguese and Italian.
Sentences between 20 and 200 characters long.

We can see an example of a sentence from each language in Table 1. Our objective is to create a model that can predict the Target Variable using the Text provided.
| Language  | Target Variable  |      text      |
| ----------| -------- | ------------------------------------------------- |
| German    | due      | Ich war um seine Gesundheit besorgt.              |
| English   | eng      | He doffed his hat when he saw me.                 |
| French    | fra      | Il a ce qu'il faut pour réussir dans le monde  des affaires.  |
| Italian   | ita      | Non lo augurerei a nessuno.                       |
| Portuguese| por      | Cante-me uma canção de ninar.                     |
| Spanish   | spa      | El surgimiento del exoesqueleto en los artrópodos  fue un acontecimiento evolutivo muy importante para esos animales |
