# Offensive Tweet Classification and Categorization 
Using the annotated OLID dataset for offensive tweets, I have used python and the deep learning framwork Keras to build the models for completing two tasks -

1) Build a classifier that classifies tweets as Offensive and Not Offensive

2. Build a classifier for Subtasks A-C combined, treating this as one single multi-class problem. There are 5 possible classes:
NOT, OFF-UNT, OFF-TIN-IND, OFF-TIN-GRP, OFF-TIN-OTH

Dataset - OLID (https://sites.google.com/site/offensevalsharedtask/olid)

Competition link - OffensEval 2019 (https://sites.google.com/site/offensevalsharedtask/offenseval2019)

<b>Task 1</b>

The testing data is split into 3 label files and 3 data files for 3 subtasks. The training data is contained in one file with all labels. I do the following steps to build the classifier and evaluate it on test data -

    •	Use pandas dataframe to read dataset and label files for training and test

    •	Read the 3 subtask labels from training dataset file into 3 different lists

    •	Start preprocessing the tweets for both training and test dataset -
    
        	Convert emojis to text
    
        	Tokenize using whitespace
    
        	Remove punctuations except some characters
    
        	Remove empty, strings, digits
    
        	Remove the word ‘user’,’url’ and ‘maga’
    
        	Remove stop words

    •	Create word to index and index to word dictionary out of all possible words in the training data

    •	Transform words in every tweet to IDs that can me mapped back to words using the dictionaries above

    •	Transform test dataset tweets to IDs using the same dictionary and assign a fixed ID to words that are not in the training dataset

    •	Encode string labels to number

    •	Pad each sentence in the pre-processed training and test dataset with zeros for a fixed length

    •	Split training data to training and validation. 3000 samples for validation and rest for training

    •	Feed prepared data into model and obtain training and validation loss and accuracies by training over a fixed number of epochs

    •	Evaluate trained model on trained dataset

<b>Task 2</b>

To build a classifier for subtasks A-C combined for 5 specified classes we have to do the following steps – 

    •	concatenate the labels for both test and training datasets. Each entry in the dataset will have one of the labels - NOT, OFF-UNT,       OFF-TIN-IND, OFF-TIN-GRP, OFF-TIN-OTH.
    
    •	Use the same pre-processed training and testing data
    
    •	Create one hot vector representation for the label data
    
    •	Change the model architecture and hyperparameters to adapt to the multi class classification problem
