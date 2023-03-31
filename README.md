# Natural Language Processing Final Project
## Text Classification
Authors: Santiago Manotas-Arroyave, Catalina Odizzio, Agostina Alexandra Pissinis, Gaurav ABS

**Contents**
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
- [Models Comparison](#modelscomparison)
- [Next Steps](#nextsteps)

## Introduction
This project aims to develop text classification models using the `IMDb dataset`. The `IMDb dataset` is a widely used benchmark in natural language processing research, containing a large collection of movie reviews classified as positive or negative. The goal of this project is to explore different machine learning algorithms, such as Logistic Regression, Recurrent Neural Networks and Bidirectional Encoder Representations from Transformers, to develop accurate text classification models that can distinguish between positive and negative movie reviews. Additionally, we will evaluate the performance of the models based on several metrics, such precision, recall, and F1 score. 

## Dataset
The dataset used in this project is the IMDb Movie Review dataset, which is available in the Hugging Face dataset library. This dataset contains a collection of movie reviews from the IMDb website, classified as positive or negative based on the rating given by the reviewer. The dataset consists of 50,000 reviews, with 25,000 reviews for training and 25,000 reviews for testing. Each review is represented as a string of characters and has a corresponding sentiment label of either positive or negative.

## Models

### Baseline

### RNN
The TensorFlow RNN model implemented is a text classification model that uses a recurrent neural network (RNN) to classify movie reviews as either positive or negative. The model uses the TensorFlow framework and Keras API to build and train a sequence model that processes text data in a sequence and outputs a binary classification. The model preprocesses the text data by tokenizing, padding, and embedding the sequences of words, and then feeds them into a bidirectional LSTM layer followed by a dense output layer with sigmoid activation function to produce the classification result. The model is trained on a labeled dataset of movie reviews, and evaluated on a test dataset to measure its accuracy.

A second RNN was established, defining a more complex model with additional layers: Two bidirectional LSTM layers are used instead of one. The first LSTM layer has 64 units and returns sequences (`return_sequences=True`); allowing the second bidirectional LSTM layer with 32 units to process the sequence output of the first layer. After this, a Dense layer with 64 neurons and sigmoid activation is added. As per the Keras classifier documentation, "using an RNN with return_sequences=True (...) makes the output still has 3-axes, like the input, so it can be passed to another RNN layer.".

Finally, metrics are calculated here directly when fitting the model, making use of the `metrics` module available from TensorFlow.

### BERT
We performed different models using `BERT` architecture. For this purpose, we used different pre-trained models, such as BERT-base, DistilBERT-base, and DistilBERT-base-uncased-finetuned-imdb. Each pre-trained model had different strengths and weaknesses that influenced the overall performance of the resulting model. By utilizing them, we were able to explore these variations and determine which models were most effective for our specific task. 

Firstly, we tested those different pre-trained models and found that the one with the best performance was `DistilBERT-base-uncased-finetuned-imdb`, which has been specifically fine-tuned on the `IMDb dataset`. Then, based on the error analysis, we adjusted different parameters of our model that allowed us to improve its performance. Some of the parameters that we tuned were the maximum sequence length for the input text, with the aim of including different lengths of text that could improve the prediction, and the dropout layer to prevent overfitting.


## Models Comparison


## Next Steps

