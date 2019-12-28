import sklearn
from sklearn.datasets import load_files
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import  PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import  classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

TRAINING_DATA_DIR = "./data/training/"
TESTING_DATA_DIR = "./data/testing/"

training_data = load_files(TRAINING_DATA_DIR, encoding="utf-8", decode_error="replace")
testing_data = load_files(TESTING_DATA_DIR, encoding="utf-8", decode_error="replace")

is_preprocessing_available = True
is_porting_available = False
is_lemmatize_available = True


#print(training_data)
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
#print((x_train)[0])
#print((y_train)[0])

#print(type(x_train))

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



def calculate_accuracy(matrix):
    print("Confusion Matrix \n")
    print(matrix)
    TP = matrix[0][0] + matrix[1][1] + matrix[2][2] + matrix[3][3]
    TN = 0
    All = matrix.sum()
    accuracy = (TP + TN) / All
    print("Accuracy is :",accuracy)
    print("\n")

if is_preprocessing_available :
    x_train = pre_processing(x_train)
    x_test = pre_processing(x_test)

#vectorizer
vectorizer = TfidfVectorizer(max_features=1000)
vectorizer.fit(x_train)
print(vectorizer.transform(x_train))

#training model MNB
MNB_Classifier = MultinomialNB()
MNB_Classifier.fit(vectorizer.transform(x_train), y_train)

y_predict = MNB_Classifier.predict(vectorizer.transform(x_test))
confusion_mat = metrics.confusion_matrix(y_test, y_predict)
print("Multinomial Nayes Bayes\n")
calculate_accuracy(confusion_mat)


## KNN
k_range = range(1,26)
knn_scores ={}
knn_scores_list =[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(vectorizer.transform(x_train), y_train)
    y_predict = knn.predict(vectorizer.transform(x_test))
    knn_scores[k] = metrics.accuracy_score(y_test,y_predict)
    knn_scores_list.append(metrics.accuracy_score(y_test,y_predict))

plt.plot(k_range,knn_scores_list)
plt.title('KNN for Text Classification')
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()



## Random Forest
rf_range = range(1,25)
rf_scores ={}
rf_scores_list =[]
for k in rf_range:
    randforest = RandomForestClassifier(n_estimators=k,random_state=0)
    randforest.fit(vectorizer.transform(x_train), y_train)
    y_predict = randforest.predict(vectorizer.transform(x_test))
    rf_scores[k] = metrics.accuracy_score(y_test,y_predict)
    rf_scores_list.append(metrics.accuracy_score(y_test,y_predict))

print(rf_scores)
print(rf_scores_list)

plt.plot(rf_range,rf_scores_list)
plt.title('RandomForest for Text Classification')
plt.xlabel('Value of Estimator for Random Forest')
plt.ylabel('Testing Accuracy')
plt.show()


##SVM
kernel =['linear','rbf']
svm_scores ={}
svm_scores_list =[]
for ker in kernel:
    svmcf = svm.SVC(kernel=ker,gamma='auto')
    svmcf.fit(vectorizer.transform(x_train), y_train)
    y_predict = svmcf.predict(vectorizer.transform(x_test))
    svm_scores[ker] = metrics.accuracy_score(y_test, y_predict)
    svm_scores_list.append(metrics.accuracy_score(y_test, y_predict))


print(svm_scores)
print(svm_scores_list)


plt.plot(kernel, svm_scores_list)
plt.title('SVM for Text Classification')
plt.xlabel('Kernel')
plt.ylabel('Testing Accuracy')
plt.show()


##Bernouli Naive Bayes
BLI_Classifier = BernoulliNB()
BLI_Classifier.fit(vectorizer.transform(x_train), y_train)
y_predict = BLI_Classifier.predict(vectorizer.transform(x_test))
confusion_mat = metrics.confusion_matrix(y_test, y_predict)
print("Bernouli Nayes Bayes\n")
calculate_accuracy(confusion_mat)



##Logistic Regression
logisticRegr = LogisticRegression(multi_class='auto',solver='saga')
logisticRegr.fit(vectorizer.transform(x_train), y_train)
y_predict = logisticRegr.predict(vectorizer.transform(x_test))
confusion_mat = metrics.confusion_matrix(y_test, y_predict)
print("Logistic Regression\n")
calculate_accuracy(confusion_mat)

##SGD Classifier
max_itr =[100,200,300,400, 500, 600, 700, 800, 900,1000]
sgd_scores ={}
sgd_scores_list =[]
for itr in max_itr:
    sgdclf = SGDClassifier(shuffle=True, max_iter=itr)
    sgdclf.fit(vectorizer.transform(x_train), y_train)
    y_predict = sgdclf.predict(vectorizer.transform(x_test))
    sgd_scores[itr] = metrics.accuracy_score(y_test, y_predict)
    sgd_scores_list.append(metrics.accuracy_score(y_test, y_predict))

#print(sgd_scores )
#print(sgd_scores_list)

plt.plot(max_itr, sgd_scores_list)
plt.title('SGD for Text Classification')
plt.xlabel('Maximum Iterations')
plt.ylabel('Testing Accuracy')
plt.show()

