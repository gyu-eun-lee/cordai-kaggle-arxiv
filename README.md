# cordai-kaggle-arxiv

The [CORD.ai arXiv Paper Classification Competition](https://www.kaggle.com/competitions/cordai-arxiv-paper-classification) ran on Kaggle from 2023-01-27 to 2023-02-13. The objective of the competition was to produce subject classifications for a dataset of arXiv preprints based on their title and summary text. This repository contains a refactored version of my code for this competition.

* `eda.ipynb` contains some exploratory data analysis for the given dataset. You can view output cells directly without running.
* `cordai_arxiv.ipynb` contains a full preprocessing/training/inference implementation. It produces model weights and output probabilities in the`./output` directory. To use the Weights & Biases logger you will need to replace the contents of the `api_key` file.
* `model.py` contains model and dataloader definitions.
* `preprocessing.py` contains methods for text preprocessing.

This README also doubles as a report of what I learned throughout the project.

# Problem description

The dataset consists of a training and testing set, both `.csv` files. Each row corresponds to a preprint from the arXiv preprint repository. The training set has the following columns:

* `ids`: preprint identification string

* `titles`: preprint title

* `summaries`: preprint summary/abstract

* `terms`: preprint's arXiv subject classifications, as a Python list converted to string

The test set has the same columns except the `terms` column. Although not explicitly stated in the competition details, the dataset for this competition was likely the same dataset used in the Keras tutorial [Large-scale multi-label text classification](https://keras.io/examples/nlp/multi_label_classification/).

The objective is to predict the target column `terms` (after converting to a multi-hot binary encoding) from the feature columns `titles` and `summaries`. In addition, the competition enforces the following restrictions:

* no use of external data.

* no use of pre-trained models.

# Challenges

The nature of this problem is a multi-label classification problem in the natural language processing domain. Over the course of the problem, I ran into the following challenges, some inherent to the problem and some due to personal limitations.

**Problem-inherent challenges:**

* *Class imbalance:* The provided dataset has a major class imbalance problem. After accounting for probable duplicate entries, the dataset contains over 30000 unique entries. There are 60 subject classes in total for the multi-label classification problem. The most populous class, `cs.CV`, applies to over 60% of all entries, whereas the least populous class contains only 3 entries (less than 0.01%). Most classes outside the top 5 apply to less than 0.1% of entries. Under such heavy class imbalance, ML models have a tendency to overfit heavily, and managing this is a key part of this problem.

* *Class sparsity:* Similar to the class imbalance problem, but along a different axis. Many classes in this dataset contain well under 100 samples, some containing fewer than 10. It is unreasonable to expect a model trained on such a dataset to learn to classify these sparse classes. Augmenting the dataset to account for this is another key aspect of the problem.

* *Competition rules:* The competition rules forbidding the use of external data and pre-trained models means that many powerful deep learning tools in NLP are unavailable. In particular, we lose access to pre-trained word embeddings and transfer learning from powerful models (especially transformer-based models).

**Personal limitations:**

* *Time and resource constraints:* The competition ran over a period of 3 weeks, and in fact I discovered the competition after it had already started. With the computing resources at my personal disposal, I found it unlikely that I would successfully train a SOTA model architecture such as a transformer-based model from scratch (in order to comply with the competition rules) while also reserving sufficient time for testing and iteration.

* *Knowledge barrier:* This was my first deep learning project after going through the Stanford CS231n material, as well as my first project in NLP. This resulted in a significant learning curve. On one axis, I needed to update my workflow from very bare-bones PyTorch/TensorFlow-based training/inference loops to something more efficient and full-featured. On the other axis, I needed to bring myself up to date on methods and models in the NLP domain. Both of these cut into the aforementioned time and resource constraints.

# Modeling decisions and outcomes

The following modeling decisions were implemented in response to the above challenges.

## Oversampling

The class imbalance and sparsity problems are likely the biggest issues to tackle. There are a number of techniques to deal with class imbalance, though some of the standard ones in data science are a bit harder to apply to NLP problems and to multi-label classification. In the end, I opted to implement an extremely crude oversampling technique to augment the training set:
* Sample N examples from each class with replacement.
* For each example, augment the textual features by randomly sampling k tokens from the tokenized text. I deemed that random sampling of tokens was acceptable for this problem as it is a classification problem; context is relatively unimportant.

Despite the unsophisticated nature of this procedure, I found that it still provided a significant boost to my chosen model's ability to generalize (without it my model was unable to generalize at all).

## Model architecture:

For this problem I wanted to see how far I could go with just convolutional and fully connected layers. For starters, fellow competitor oldjerry had already submitted a very nice [competition notebook](https://www.kaggle.com/code/oldjerry/xgb-tab-rf-lr-fusion-based-on-multi-method) based on ensembling several models from scikit-learn, and I wanted to see what could be done in a different direction.

For a while I experimented with implementing LSTMs and transformers, but eventually concluded that the training time and computing resources required to train such models from scratch would not be an effective use of the time left in the competition. Therefore I opted for a simpler model, at first a very simple fully connected NN before arriving at a wide convolutional NN proposed in [Convolutional Neural Networks for Sentence Classification](https://aclanthology.org/D14-1181) (Kim, EMNLP 2014). This model outperformed my basic fully connected models with a similar number of parameters, while being fairly quick to train, so eventually this became my model architecture of choice. To summarize the model:

* A preprocessing step is applied in which a vocabulary is trained on the textual corpus comprising the training data, and then used to encode text samples into integer vectors.
* The initial layer is an embedding layer. Given the competition rules, this layer also needs to be trained from scratch rather than from an existing word embedding.
* The next layer consists of 3 separate convolutional layers side-by-side, with filter sizes 1, 3, and 5 respectively. In principle, these layers should learn local features of textual data broken down as n-grams for n = 1, 3, and 5. (There was no data-driven reason for the choices 1, 3, and 5: these were chosen because they make it possible to choose a padding length that preserves the size of the input, allowing for an easy implementation of a residual connection.)
* Each output from the convolutional layer then goes through a pooling layer, implemented as a convolutional layer with filter size 2 and stride 2. In principle this should also help the model learn about n-gram features for other values of n.
* The outputs are concatenated, then sent through one or more fully connected layers until a final 60-dimensional output (for the 60 possible subject classes) is produced.
* Outputs are converted to probabilities, and the model is optimized for binary cross-entropy loss.

Overall, with the crude oversampling augmentation and this model architecture, a version of this model with ~32.5M parameters was able to produce a validation cross-entropy loss of ~0.323 and reasonably strong validation metrics (e.g. 0.883 AUC ROC score) in just 6 epochs of training with 4GB GPU RAM. Stronger performance was observed with greater batch size and larger augmented dataset size on the Kaggle kernel (~0.198 cross-entropy loss). This is far from ideal (the top competition score reports a cross-entropy loss of ~0.065), but seems decent enough for this architecture.

# Observations during model selection:

* The data augmentations/dataloader constructions downsampled the textual data: most texts have on the order of 1e2 tokens, but post-processing (random sampling + padding) would become randomly sampled inputs with far fewer tokens (32 in the present architecture). I found this not to hurt the model's effectiveness, and in fact the model benefited from the downsampling when compared to processing into much longer length inputs: Bayesian hyperparameter tuning using Weights & Biases revealed a moderate anti-correlation between input text length and validation metrics.

* W&B tuning suggests that embedding dimension is one of the more important model parameters up to a point, with at least a moderate positive correlation to validation metrics.

* Vocabulary size, on the other hand, was seen to be less important but with moderate anti-correlation. This could be because the arXiv text corpus is likely to contain many technical and rarely-used words that often are highly indicative of a subject classification. It is possible that restricting the vocabulary size during vocabulary construction (with TF-IDF weighting) helps to narrow down to these technical words.

* Adding an extra wide convolutional layer didn't seem to be too useful for this architecture.

# Improving the model

The final version of the model only really came together in the final few days of the competition, so there is plenty to be done. Barring a completely different choice of model architecture, the following are potential ways to improve the model's performance:

* By far the most important is probably better data preprocessing and augmentation. The oversampling given here is extremely crude and ends up losing a lot of information about word order and local textual features. The convnet is designed in a way that lets it learn local textual features, so there is a mismatch between the modeling decisions in the augmentation step and the model selection step. Given more time, I would have liked to implement an augmentation method that preserves some local textual features.

* Similarly, the text vectorization ended up being very simple (vocabulary training on training corpus with TF-IDF weighting); due to the random sampling augmentation, there was no point in generating a vocabulary with bigrams or trigrams. Again, a more sophisticated local-feature-preserving augmentation would have allowed us to vectorize with bigrams and trigrams, which I suspect are extremely important features for this type of classification task on technically dense text.

* There is plenty of room for fine-tuning the layer parameters - I chose layer dimensions as powers of 2 for convenience's sake during experimentation. I don't think this will necessarily result in the biggest gains, as this model is more like to reach an upper bound to its representational power.

* Originally I wanted to handle the multilabel classification problem by classifier chaining: training a binary classification problem on a single class, then using the outputs as input into a binary classification problem for the next class, and so on. I didn't have time before the competition's end to see how this works out, but it would be an interesting experiment to see if this outperforms classifying all classes at once. That said, some EDA reveals that most classes are uncorrelated with each other (aside from the 5 most populous classes, and some class pairs which are perfectly correlated), so I would expect that this wouldn't add much additional information for the model to work with.

# Learning outcomes
Here are some of the things I learned throughout the course of this project.

## Modeling
* Data augmentation should be a priority. I began implementing data augmentation very late into the project, but this is where I saw the biggest gains even with the extremely crude implementation given here. No amount of fiddling with model architecture will address structural deficiencies in the data.
* Model selection should go from simple to complicated, not vice versa. I wasted a lot of time earlier on exploring complex models like LSTMs and transformers, when in fact it would have been more effective to start off with a very simple model like a shallow fully connected network and devote more time to finding good data augmentations instead. In fact, for a problem like this one where context is likely to be relatively unimportant, it is almost definitely a mistake to jump to a complicated architecture first.
* Tuning model architecture parameters and hyperparameters is tricky business. Even with automated hyperparameter search, it is probably wise to work on a few parameters at a time.

## Implementation
Arguably my biggest takeaway from this project is code maturity. At the beginning of this project, I was working essentially in base PyTorch/TensorFlow with all of the associated overhead and boilerplate code. Since then, I have made the following updates to my ability to implement and train deep learning models:

* moving to PyTorch Lightning, allowing me to focus more on desired training outcomes rather than the minutiae of manually coding training loops, while also taking advantage of well-optimized training loops and callback code.

* adopting Weights & Biases automated hyperparameter search and visualization.

* learning how to implement basic objectives in NLP such as text vectorization, tokenization, embedding, as well as more advanced objectives like LSTMs and transformers via the PyTorch and HuggingFace libraries.