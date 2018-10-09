# Formality Classifier

This is a basic machine learning model that will classify sentences as either formal or informal. This is only a start for this project, more work needs to be done to improve on the model's performance. Currently the training data is only from two sources. Reddit comments were used for informal text examples, and the Brown Corpus was used for formal samples primarily because it's just built in to NLTK. 

## Usage
Training a machine learning model takes a lot of computational resources, it is recommended that you install the GPU version of tensorflow if you have one. It will make training go up to 50 times faster.

There are two main components to this project: training and testing. 

* `train.py` will load all of the training data, and then start training the model. This will take a long time, several minutes to an hour.
* `test_classifier.py` will load the pre-trained model and allow you to test out sentences to see how the classifier performs.

In order to train the model you'll need to download the [reddit comments dataset](https://drive.google.com/a/g.rit.edu/file/d/1h4u1PVSfc3GKxl0K_sokYKdlzMNbbs11/view?usp=sharing) and the [glove6B word vectors](http://nlp.stanford.edu/data/glove.6B.zip). Once you have those downloaded, unzip both of them. Create a directory in the root of this project called data with the following structure, using the files you just downloaded.


```
data
├───glove.6B
│   └───glove.6B.100d.txt
└───reddit_comments.txt
```

If you just want to test out the classifier, you can easily do that by running the `test_classifier.py` script. It will automatically load the model trained on 10,000 sentences and allow you to easily classify any sentence you type in. It will also classify a set of sentences built into the program so you can immediately see how well it performs.
