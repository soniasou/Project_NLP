# Project_NLP

## Table of contents

- [Introduction](#introduction)
- [Bag of Word Technique](#bag-of-word-technique)
- [Glove Technique](#glove-technique)
- [Running Project](#Running-Project)

## Introduction

The project consists in building a recommender system based on movie contents. Two techniques will be used: Bag of word technique and Glove technique. The first step in consdering both techniques is to calculate embeddings for each technique. Then, we have to build an annoy index for each embedding. Finally, the main objective will be to return 5 most similar movies according to the description you give and the embeddings you calculated. The database is founded here: 


## Bag of Word Technique

This technique consists in representing a document as an unordered set of words, neglecting order and grammatical structure, but retaining information on word frequency. The first step here consists in calculating embeddings based on bag of word and movie reviews. The 'bag of word' embedding calculation function consists of several steps: 

- Importing the list of empty words in English.

- Defining a tokenizer with stemming to split the text into words, convert to lower case and simplify to roots. 

- Initializing tokenizer.

- Creating a list of processed stop words. 

- Extracting embeddings from text: Transform text into a numerical vector, then use the CounVectorizer method to transform text into a vector.  The vector is then reshaped to have a fixed length of 1000. The dataset 'movies_data' in the Drive link contains the column of 'embeddings_bag_of_words' for this method. Here is the link to the database:
  
https://drive.google.com/file/d/1fih6eesBm8_DPSYn1X31-kesEyHEbdH6/view?usp=drive_link

The next step is then generating the annoy index 'index.ann':

https://drive.google.com/file/d/1eumdZ-4BUsgVfMK0jw98PRJeC2hgfQR_/view?usp=drive_link

## Glove Technique

The glove technique aims to capture the semantic meaning of words by representing them as vectors in a vector space of fixed dimensions. The 'glove' embedding calculation function consists of several steps: 

- Loading pre-trained GloVe word embeddings using the gensim library. It converts the GloVe format to the Word2Vec format, and then loads the embeddings into a KeyedVectors model.For this purpose, we use the following file 'glove.6B.100d.txt':

https://drive.google.com/file/d/1Y2p5dtqRW6WK0CBAIHicrEaMFWBwePsf/view?usp=drive_link

- Computing text embedding by converting the text into lowercase words, retrieving the embeddings for each word from the GloVe model, and then calculating the average of these embeddings to obtain a single vector representation for the entire text. Vectors generated are of length 100. The same dataset contains the column of 'embeddings_glove' for this method.

The next step is then generating the annoy index 'index_1.ann':

https://drive.google.com/file/d/1tI2DIdsVKqrI-7DsXGHvZSLw06hzjqfg/view?usp=drive_link

## Running project

The first step in this project is running the Docker compose:

```bash
docker-compose up
```

Then open `localhost:7860`, enter your movie description, and choose whether you want to generate the most 5 similar movies based on their description using 'Bag_of_word' or 'Glove'. 



