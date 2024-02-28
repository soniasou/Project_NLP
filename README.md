# Project_NLP

## Table of contents

- [Introduction](#introduction)
- [Bag of Word Technique](#bag-of-word-technique)
- [Glove Technique](#glove-technique)
- [Running Project](#running-project)

## Introduction

The project consists in building a recommender system based on movie contents. Two techniques will be used: Bag of word technique and Glove technique. The first step in consdering both techniques is to calculate embeddings for each technique. Then, we have to build an annoy index for each embedding. Finally, the main objective will be to return 5 most similar movies according to the description you give and the embeddings you calculated. The database is founded here: 


## Bag of Word Technique

This technique consists in representing a document as an unordered set of words, neglecting order and grammatical structure, but retaining information on word frequency. The first step here consists in calculating embeddings based on bag of word and movie reviews. The 'bag of word' embedding calculation function consists of several steps: 

- Importing the list of empty words in English.

- Defining a tokenizer with stemming to split the text into words, convert to lower case and simplify to roots. 

- Initializing tokenizer.

- Creating a list of processed stop words. 

- Extracting embeddings from text: Transform text into a numerical vector, then use the CounVectorizer method to transform text into a vector.  The vector is then reshaped to have a fixed length of 1000.

## Glove Technique
