import csv
import itertools as it
import numpy as np
np.random.seed(0)
import lab_util
import sklearn.decomposition as skd

'''
## Introduction

In this lab, you'll explore three different ways of using unlabeled text data to learn pretrained word representations. 
Your lab report will describe the effects of different modeling decisions (representation learning objective, 
context size, etc.) on both qualitative properties of learned representations and their effect on a downstream prediction problem.

**General lab report guidelines**
Homework assignments should be submitted in the form of a research report. (We'll be providing a place to upload them 
before the due date, but are still sorting out some logistics.) Please upload PDFs, with a maximum of four 
single-spaced pages. (If you want you can use the [Association for Computational Linguistics style files]
(http://acl2020.org/downloads/acl2020-templates.zip).) Reports should have one section for each part of the homework 
assignment below. Each section should describe the details of your code implementation, and include whatever charts / 
tables are necessary to answer the set of questions at the end of the corresponding homework part.

We're going to be working with a dataset of product reviews. It looks like this:
'''
# ----------------------------------------------------------------------------------------------------------------------
# Part 1: Initial Data
# ----------------------------------------------------------------------------------------------------------------------

data = []
n_positive = 0
n_disp = 0
with open("./reviews.csv") as reader:
    csvreader = csv.reader(reader)
    next(csvreader)
    for id, review, label in csvreader:
        label = int(label)
    # hacky class balancing
        if label == 1:
            if n_positive == 2000:
                continue
            n_positive += 1
        if len(data) == 4000:
            break
        data.append((review, label))
    
        if n_disp > 5:
            continue
        n_disp += 1
        # print("review:", review)
        # print("rating:", label, "(good)" if label == 1 else "(bad)")
        # print()

# print(f"Read {len(data)} total reviews.")
np.random.shuffle(data)
reviews, labels = zip(*data)
train_reviews = reviews[:3000]
train_labels = labels[:3000]
val_reviews = reviews[3000:3500]
val_labels = labels[3000:3500]
test_reviews = reviews[3500:]
test_labels = labels[3500:]

# ----------------------------------------------------------------------------------------------------------------------
# Part 1: Word representations via matrix factorization
# ----------------------------------------------------------------------------------------------------------------------
'''
First, we'll construct the term--document matrix (look at `/content/6864-hw1/lab_util.py` in the file browser on the 
left if you want to see how this works).
'''

vectorizer = lab_util.CountVectorizer()
vectorizer.fit(train_reviews)
td_matrix = vectorizer.transform(train_reviews).T
# print(f"TD matrix is {td_matrix.shape[0]} x {td_matrix.shape[1]}")  # (2006, 3000)

'''
First, implement a function that computes word representations via latent semantic analysis:
'''

def learn_reps_lsa(matrix, rep_size):
    # `matrix` is a `|V| x n` matrix, where `|V|` is the number of words in the
    # vocabulary. This function should return a `|V| x rep_size` matrix with each
    # row corresponding to a word representation. The `sklearn.decomposition`
    # package may be useful.
    svd = skd.TruncatedSVD(n_components=rep_size)
    svd.fit(matrix.T)
    output = svd.components_
    return output.T

"""Let's look at some representations:"""

reps = learn_reps_lsa(td_matrix.copy(), 500)   # (n_word, rep_size)
# words = ["good", "bad", "cookie", "jelly", "dog", "the", "4"]
words = ["good", "dog", "the", "3"]
show_tokens = [vectorizer.tokenizer.word_to_token[word] for word in words]
# lab_util.show_similar_words(vectorizer.tokenizer, reps, show_tokens)

'''
We've been operating on the raw count matrix, but in class we discussed several reweighting schemes aimed at making LSA 
representations more informative. 

Here, implement the TF-IDF transform and see how it affects learned representations.
'''

def transform_tfidf(matrix):
    # `matrix` is a `|V| x |D|` matrix of raw counts, where `|V|` is the
    # vocabulary size and `|D|` is the number of documents in the corpus. This
    # function should (nondestructively) return a version of `matrix` with the
    # TF-IDF transform appliied.
    # matrix = td_matrix.copy()
    tfidf = np.zeros(matrix.shape)
    n_word = matrix.shape[0]
    n_doc = matrix.shape[1]
    for i in range(n_word):
        i_doc = matrix[i, :]
        i_doc[i_doc > 0] = 1
        n_doc_i = i_doc.sum()
        tfidf[i, :] = matrix[i, :] * np.log(n_doc / n_doc_i)
    return tfidf

"""How does this change the learned similarity function?"""

td_matrix_tfidf = transform_tfidf(td_matrix.copy())
reps_tfidf = learn_reps_lsa(td_matrix_tfidf, 750)
# lab_util.show_similar_words(vectorizer.tokenizer, reps_tfidf, show_tokens)

"""Now that we have some representations, let's see if we can do something useful with them.

Below, implement a feature function that represents a document as the sum of its
learned word embeddings.

The remaining code trains a logistic regression model on a set of *labeled* reviews; 
we're interested in seeing how much representations learned from *unlabeled* reviews improve classification.
"""

def word_featurizer(xs):
  # normalize
    return xs / np.sqrt((xs ** 2).sum(axis=1, keepdims=True))

def lsa_featurizer(xs):
  # This function takes in a matrix in which each row contains the word counts
  # for the given review. It should return a matrix in which each row contains
  # the learned feature representation of each review (e.g. the sum of LSA 
  # word representations).
  feats = np.dot(xs, reps_tfidf)  # Your code here!
  # normalize
  return feats / np.sqrt((feats ** 2).sum(axis=1, keepdims=True))

def combo_featurizer(xs):
  return np.concatenate((word_featurizer(xs), lsa_featurizer(xs)), axis=1)

def train_model(featurizer, xs, ys):
  import sklearn.linear_model
  xs_featurized = featurizer(xs)
  model = sklearn.linear_model.LogisticRegression()
  model.fit(xs_featurized, ys)
  return model

def eval_model(model, featurizer, xs, ys):
  xs_featurized = featurizer(xs)
  pred_ys = model.predict(xs_featurized)
  test_acc = np.mean(pred_ys == ys)
  print("test accuracy", test_acc)
  return test_acc

def training_experiment(name, featurizer, n_train):
  print(f"{name} features, {n_train} examples")
  train_xs = vectorizer.transform(train_reviews[:n_train])
  train_ys = train_labels[:n_train]
  test_xs = vectorizer.transform(test_reviews)
  test_ys = test_labels
  model = train_model(featurizer, train_xs, train_ys)
  test_result = eval_model(model, featurizer, test_xs, test_ys)
  print()
  return test_result

# training_experiment("word", word_featurizer, 10)
# training_experiment("lsa", lsa_featurizer, 10)
# training_experiment("combo", combo_featurizer, 10)

import pandas as pd
train_num = [10] + [i*100 for i in range(1, 31)]
word_test = []
lsa_test = []
combo_test = []
for n_train in train_num:
    word_test.append(training_experiment("word", word_featurizer, n_train))
    lsa_test.append(training_experiment("lsa", lsa_featurizer, n_train))
    combo_test.append(training_experiment("combo", combo_featurizer, n_train))
test_eval = [word_test, lsa_test, combo_test]
test_eval = np.array(test_eval)
test_eval_df = pd.DataFrame(test_eval.T, columns=['word', 'lsa', 'combo'], index=train_num)
test_eval_df.to_csv('./test_eval_750.csv')


"""**Part 1: Lab writeup**

Part 1 of your lab report should discuss any implementation details that were important to filling out the code above. 
Then, use the code to set up experiments that answer the following questions:

1. Qualitatively, what do you observe about nearest neighbors in representation space? 
(E.g. what words are most similar to _the_, _dog_, _3_, and _good_?)

2. How does the size of the LSA representation affect this behavior?

3. Recall that the we can compute the word co-occurrence matrix $W_{tt} = W_    
   {td} W_{td}^\top$. What can you prove about the relationship between the    
   left singular vectors of $W_{td}$ and $W_{tt}$? Do you observe this behavior 
   with your implementation of `learn_reps_lsa`? Why or why not?

4. Do learned representations help with the review classification problem? What
   is the relationship between the number of labeled examples and the effect of
   word embeddings?
   
5. What is the relationship between the size of the word embeddings and their usefulness for the classification task.
"""

# ----------------------------------------------------------------------------------------------------------------------
# Part 2: Word representations via language modeling
# ----------------------------------------------------------------------------------------------------------------------
"""
In this section, we'll train a word embedding model with a word2vec-style objective rather than a matrix 
factorization objective. This requires a little more work; we've provided scaffolding for a PyTorch model 
implementation below.
(If you've never used PyTorch before, there are some tutorials [here](https://pytorch.org/tutorials/). 
You're also welcome to implement these experiments in any other framework of your choosing.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torch_data

class Word2VecModel(nn.Module):
    # A torch module implementing a word2vec predictor. The `forward` function
    # should take a batch of context word ids as input and predict the word
    # in the middle of the context as output, as in the CBOW model from lecture.

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.linear1 = nn.Linear(2*2*embed_dim, 64)
        self.linear2 = nn.Linear(64, vocab_size)

    def forward(self, context):
        # Context is an `n_batch x n_context` matrix of integer word ids
        # this function should return a set of scores for predicting the word
        # in the middle of the context
        embedded = self.embeddings(context)
        embedded = embedded.view(-1, self.num_flat_features(embedded))
        hid = F.relu(self.linear1(embedded))
        out = self.linear2(hid)
        log_probs = F.log_softmax(out)
        return log_probs

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def learn_reps_word2vec(corpus, window_size, rep_size, n_epochs, n_batch):
    # This method takes in a corpus of training sentences. It returns a matrix of
    # word embeddings with the same structure as used in the previous section of
    # the assignment. (You can extract this matrix from the parameters of the
    # Word2VecModel.)

    tokenizer = lab_util.Tokenizer()
    tokenizer.fit(corpus)
    tokenized_corpus = tokenizer.tokenize(corpus)

    ngrams = lab_util.get_ngrams(tokenized_corpus, window_size)

    # device = torch.device('cuda')  # run on colab gpu
    # model = Word2VecModel(tokenizer.vocab_size, rep_size).to(device)
    model = Word2VecModel(tokenizer.vocab_size, rep_size)
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.NLLLoss()

    loader = torch_data.DataLoader(ngrams, batch_size=n_batch, shuffle=True)

    for epoch in range(n_epochs):
        for context, label in loader:
            # as described above, `context` is a batch of context word ids, and
            # `label` is a batch of predicted word labels
            # Your code here!
            model.zero_grad()
            log_probs = model(context)
            loss = loss_fn(log_probs, label)

            loss.backward()
            opt.step()
    # reminder: you want to return a `vocab_size x embedding_size` numpy array
    embedding_matrix = model.embeddings.weight
    embedding_matrix = embedding_matrix.detach().numpy()
    return embedding_matrix


# corpus = train_reviews
# window_size, rep_size, n_epochs, n_batch = 2, 500, 10, 100
reps_word2vec = learn_reps_word2vec(train_reviews, 2, 500, 10, 100)

"""After training the embeddings, we can try to visualize the embedding space to see if it makes sense. 
First, we can take any word in the space and check its closest neighbors."""

lab_util.show_similar_words(vectorizer.tokenizer, reps_word2vec, show_tokens)

"""We can also cluster the embedding space. Clustering in 4 or more dimensions is hard to visualize, 
and even clustering in 2 or 3 can be difficult because there are so many words in the vocabulary. 
One thing we can try to do is assign cluster labels and qualitiatively look for an underlying pattern in the clusters."""

# from sklearn.cluster import KMeans
#
# indices = KMeans(n_clusters=10).fit_predict(reps_word2vec)
# zipped = list(zip(range(vectorizer.tokenizer.vocab_size), indices))
# np.random.shuffle(zipped)
# zipped = zipped[:100]
# zipped = sorted(zipped, key=lambda x: x[1])
# for token, cluster_idx in zipped:
#     word = vectorizer.tokenizer.token_to_word[token]
#     print(f"{word}: {cluster_idx}")

"""Finally, we can use the trained word embeddings to construct vector representations of full reviews. 
One common approach is to simply average all the word embeddings in the review to create an overall embedding. 
Implement the transform function in Word2VecFeaturizer to do this."""


def lsa_featurizer(xs):
    # print(xs.shape)
    feats = np.dot(xs, reps_word2vec)  # Your code here!
    # normalize
    return feats / np.sqrt((feats ** 2).sum(axis=1, keepdims=True))


training_experiment("word2vec", lsa_featurizer, 10)


"""**Part 2: Lab writeup**

Part 2 of your lab report should discuss any implementation details that were important to filling out the code above. 
Then, use the code to set up experiments that answer the following questions:

1. Qualitatively, what do you observe about nearest neighbors in representation space? 
(E.g. what words are most similar to _the_, _dog_, _3_, and _good_?)
How well do word2vec representations correspond to your intuitions about word similarity?

2. One important parameter in word2vec-style models is context size. 
How does changing the context size affect the kinds of representations that are learned?

3. How do results on the downstream classification problem compare to part 1?

4. What are some advantages and disadvantages of learned embedding representations, relative to the featurization done in part 1?

5. What are some potential problems with constructing a representation of the review by averaging the embeddings of the individual words?
"""


