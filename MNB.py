import string

import sklearn
from sklearn.datasets import load_files
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import  PorterStemmer

TRAINING_DATA_DIR = "./data/training/"
TESTING_DATA_DIR = "./data/testing/"

training_data = load_files(TRAINING_DATA_DIR, encoding="utf-8", decode_error="replace")
testing_data = load_files(TESTING_DATA_DIR, encoding="utf-8", decode_error="replace")

# calculate count articles for each category
training_lables, training_counts = np.unique(training_data.target, return_counts=True)
training_lables_str = np.array(training_data.target_names)[training_lables]
#print(dict(zip(training_lables_str,training_counts)))

testing_lables, testing_counts = np.unique(testing_data.target, return_counts=True)
testing_lables_str = np.array(testing_data.target_names)[testing_lables]
#print(dict(zip(testing_lables_str, testing_counts)))

#extract data and lables
x_train, y_train = training_data.data, training_data.target
x_test, y_test = testing_data.data, testing_data.target

is_preprocessing_available = True
is_porting_available = False
is_lemmatize_available = True

def pre_processing(list_of_data):
    pre_processed_data = []

    for single_data in list_of_data:

        tokens = word_tokenize(single_data)
        words = [word for word in tokens if word.isalpha()]

        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]

        if is_porting_available:
            porter = PorterStemmer()
            stemmed_words = [porter.stem(word) for word in words]
            stemmed_words_list = []
            stemmed_words_list = ' '.join(stemmed_words)
            pre_processed_data.append(stemmed_words_list)

        elif is_lemmatize_available:
            wordnet_lemmatizer = WordNetLemmatizer()
            lemmatized_text = []

            for word in words:
                lemmatized_text.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

            lemmatized_text =' '.join(lemmatized_text)
            pre_processed_data.append(lemmatized_text)


    return pre_processed_data

if is_preprocessing_available :
    x_train = pre_processing(x_train)
    x_test = pre_processing(x_test)

#find the vacabluary
vocabulary = {}
for i in range(len(x_train)):
    for word in x_train[i].split():
        #word_new  = word.strip(string.punctuation).lower()
        if word in vocabulary:
                vocabulary[word]+=1
        else:
                vocabulary[word]=1
#print(vocabulary.values())

#print(vocab)
num_words = [0 for i in range(max(vocabulary.values())+1)]
#print(max(vocabulary.values()))

#remove less frequency words
cutoff_freq = 1
num_words_above_cutoff = len(vocabulary)-sum(num_words[0:cutoff_freq])


features = []
for key in vocabulary:
    if vocabulary[key] >=cutoff_freq:
        features.append(key)
#print(features)

word_list =[]
X_train_dataset = np.zeros((len(x_train),len(features)))
for i in range(len(x_train)):
    word_list = [ word.strip(string.punctuation).lower() for word in x_train[i].split()]
    for word in word_list:
        if word in features:
            X_train_dataset[i][features.index(word)] += 1
#print(X_train_dataset)


X_test_dataset = np.zeros((len(x_test),len(features)))
for i in range(len(x_test)):
    word_list = [ word.strip(string.punctuation).lower() for word in x_test[i].split()]
    for word in word_list:
        if word in features:
            X_test_dataset[i][features.index(word)] += 1
#print(X_test_dataset)


def calculate_accuracy(matrix):
    print("Confusion Matrix \n")
    print(matrix)
    TP = matrix[0][0] + matrix[1][1] + matrix[2][2] + matrix[3][3]
    TN = 0
    All = matrix.sum()
    accuracy = (TP + TN) / All
    print("Accuracy is :",accuracy)
    print("\n")


class MultinomialNaiveBayes:

    def __init__(self):
        self.count = {}
        self.classes = None

    def fit(self, X_train, Y_train):
        self.classes = set(Y_train)
        for class_ in self.classes:
            self.count[class_] = {}
            for i in range(len(X_train[0])):
                self.count[class_][i] = 0
            self.count[class_]['total'] = 0
            self.count[class_]['total_points'] = 0
        self.count['total_points'] = len(X_train)
        #print(self.count)

        for i in range(len(X_train)):
            for j in range(len(X_train[0])):
                self.count[Y_train[i]][j] += X_train[i][j]
                self.count[Y_train[i]]['total'] += X_train[i][j]
            self.count[Y_train[i]]['total_points'] += 1

    def __probability(self, test_point, class_):

        log_prob = np.log(self.count[class_]['total_points']) - np.log(self.count['total_points'])
        total_words = len(test_point)
        for i in range(len(test_point)):
            current_word_prob = test_point[i] * (
                        np.log(self.count[class_][i] + 1) - np.log(self.count[class_]['total'] + total_words))
            log_prob += current_word_prob
        return log_prob

    def __predictSinglePoint(self, test_point):

        best_class = None
        best_prob = None
        first_run = True

        for class_ in self.classes:
            log_probability_current_class = self.__probability(test_point, class_)
            if (first_run) or (log_probability_current_class > best_prob):
                best_class = class_
                best_prob = log_probability_current_class
                first_run = False
        return best_class

    def predict(self, X_test):
        print(len(X_test))
        Y_pred = []
        for i in range(len(X_test)):
            Y_pred.append(self.__predictSinglePoint(X_test[i]))
        return Y_pred



myMNB_classifier = MultinomialNaiveBayes()
myMNB_classifier.fit(X_train_dataset,y_train)
Y_test_pred = myMNB_classifier.predict(X_test_dataset)
confusion_mat = metrics.confusion_matrix(y_test, Y_test_pred)
print("Multinomial Nayes Bayes\n")
calculate_accuracy(confusion_mat)
