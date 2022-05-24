import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
#=================================================================================
#-------------------------------Reading File-------------------------------------
#=================================================================================
mushroomData = pd.read_csv("D:\\3rd year\\Second term\\Data mining\\project\\mushrooms 2.csv")

#=================================================================================
#-------------------------------Preprocessing-------------------------------------
#=================================================================================

# (1) Replace '?' with nan
mushroomData.replace('?', np.NaN, inplace=True)                                                                 #feautre 11 contains '?' instead of NaN, convert it to be able to drop it

# (2) Remove columns with missing values above a certain threshold 20%       "stalk-root"
null_Percentage = mushroomData.isna().sum() / len(mushroomData)                                                #the percentage of null values in each col
toBe_removed = mushroomData.loc[:, null_Percentage >= .3]                                                       #KEEP ONLY col with percentage < 20%
mushroomData = mushroomData.drop(toBe_removed, axis=1)

# (3) SAMPLING
mushroomData = mushroomData.groupby('class', group_keys=False)
mushroomData = mushroomData.apply(lambda x: x.sample(frac=0.5))

# (4) Remove col with all unique (we don't have all unique) / all similar values(will drop "veil-type" column)  (21)
for col in mushroomData.columns:
    if len(mushroomData[col].unique()) == 1:
        mushroomData.drop(col, inplace=True, axis=1)
    elif len(mushroomData[col].unique()) == len(mushroomData):
        mushroomData.drop(col, inplace=True, axis=1)

# (5) Remove duplicates
mushroomData.drop_duplicates(inplace=True)

# (6) data splitting
Y = mushroomData['class']
X = mushroomData.drop(['class'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)                     #random state = integer mean for integer number of runs it will split in the same way+will be shuffled , shuffle= True just in case data was ordered in a certain manner (ex:- for a certain boolean feature all T values are in the start of the data set
colNames = X.columns

# (7) Encoding, encode train and test data independently to avoid data leakage
OE = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)                                      #'use_encoded_value' = lw dkhalo f feature mo3yna value/category mashfhosh abl kda during training phase will be set to -1 (value given to 'unknown_value' parameter)
x_train = pd.DataFrame(OE.fit_transform(x_train))                                                              #convert x_train & x_test to dataframe as they were originally as OE returns them as numpy array
x_test = pd.DataFrame(OE.transform(x_test))                                                                    #fit_transform with training data , transform only with testing data
y_train = OE.fit_transform(pd.DataFrame(y_train))                                                              #OE takes dataframes while y_test & y_train are series , OE will return them as 2D np array
y_test = OE.transform(pd.DataFrame(y_test))
y_train = pd.Series(y_train.flatten())                                                                         #return y_train & y_test to series objects ay they were/ .flatten() because pd.seres() take 1D array not 2D
y_test = pd.Series(y_test.flatten())

# (8) feature selection according to feature importance using ExtraTreesClassifier
# (8.1)
X = OE.fit_transform(X)
Y = pd.Series((OE.fit_transform(pd.DataFrame(Y))).flatten())
ETF_classifier = ExtraTreesClassifier(n_estimators=100, criterion='entropy', max_features= 11)                  #n_estimators = number of decision trees,  criterion = function used to measure information gain (eg: gini index, entropy) maximum number of features
ETF_classifier.fit(X, Y)
# (8.2) plotting importance of each feature
featureImp = ETF_classifier.feature_importances_
featureImp_DF = pd.DataFrame({'feature': list(colNames), 'importance': featureImp}).sort_values('importance', ascending=False)
featureImp_DF.plot(x='feature', y='importance', kind='bar')
plt.show()
# (8.3) keep only top 11 most imp features
to_drop = featureImp_DF.iloc[11:].index
x_train.drop(to_drop, inplace=True, axis=1)
x_test.drop(to_drop, inplace=True, axis=1)


#=================================================================================
#------------------------- Model Evaluation Function -----------------------------
#=================================================================================
def evaluate(pred, classifier):
    confusionM = confusion_matrix(y_test, pred)
    confusionM_DF = pd.DataFrame(data=confusionM, columns=['Actual Positive', 'Actual Negative'],index=['Predict Positive', 'Predict Negative'])
    confusionM_DF['Total'] = [confusionM[0, 1]+confusionM[0, 0], confusionM[1, 0]+confusionM[1, 1]]
    confusionM_DF.loc['Total'] = [confusionM[0, 0]+confusionM[1, 0], confusionM[0, 1]+confusionM[1, 1], confusionM[0, 0]+confusionM[1, 0]+ confusionM[0, 1]+confusionM[1, 1]]
    print('\n', confusionM_DF)
    print('\nTrue Positives(TP) = ', confusionM[0, 0])
    print('True Negatives(TN) = ', confusionM[1, 1])
    print('False Positives(FP) = ', confusionM[0, 1])
    print('False Negatives(FN) = ', confusionM[1, 0])
    print('\n****\n')
    print("Accuracy: ", accuracy_score(y_test, pred))                               # (TP+TN) / P+N
    print("Mean absolute error: ", mean_absolute_error(y_test, pred))               # where i range from 0 to n [ (Xi-Yi) /n ]   , Yi=prediction,  Xi=true value total number of data points, n = number of samples
    print("Precision score: ", precision_score(y_test, pred))                       # (TP) / (TP + FP)
    print("Recall score: ", recall_score(y_test, pred))                             # TP / P
    print("___")
    print("Check for Overfitting or Underfitting: ")  # if training score > testing score = overfitting
    print("Training set score:", classifier.score(x_train, y_train))
    print("Test set score:", classifier.score(x_test, y_test))

#=================================================================================
#------------------------------- Classifiers -------------------------------------
#=================================================================================
results = []
names = ["Random Forest","LOGISTIC REGRESSION","Decision Tree","KNN","SVM","Naive Bayes"]
print('\n################## Random Forest ######################')                                                   #Random forest generate multiple decision tree then takes the avg of all the predictions made by the trees

RF_classifier = RandomForestClassifier(n_estimators=100, random_state=100,  criterion='entropy', min_samples_leaf=50)      #n_estimators = number of decision trees in the forest
RF_classifier.fit(x_train, y_train)
y_predicted = RF_classifier.predict(x_test)
evaluate(y_predicted, RF_classifier)
results.append(accuracy_score(y_test, y_predicted))

print('\n\n############### LOGISTIC REGRESSION ##################')
LR_classifier = LogisticRegression(solver='sag', random_state=100)                                                #random_state to shuffle data, use 'sag' or 'saga' solver as they're faster for large datasets
LR_classifier.fit(x_train, y_train)
y_predicted = LR_classifier.predict(x_test)
evaluate(y_predicted, LR_classifier)
results.append(accuracy_score(y_test, y_predicted))


print('\n\n###################### KNN ##########################')   #generaly for large dataset, k should have large value ( square root N, where N is the number of neighbours)
test_score = []
train_score =[]
K_possibleValues = np.arange(3,67)
for i in K_possibleValues:
    KNNclassifier = KNeighborsClassifier(n_neighbors=i, p=5)
    KNNclassifier.fit(x_train, y_train)
    train_score.append(KNNclassifier.score(x_train, y_train))
    test_score.append(KNNclassifier.score(x_test, y_test))

plt.title('KNN with varying number of neighbors')
plt.plot(K_possibleValues, test_score, label='Testing Accuracy')
plt.plot(K_possibleValues, train_score, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()
print("Choosing K as root(N) where N is number of Samples = ", 67)
KNNclassifier = KNeighborsClassifier(n_neighbors=67)
KNNclassifier.fit(x_train, y_train)
y_pred = KNNclassifier.predict(x_test)
evaluate(y_pred, KNNclassifier)
results.append(accuracy_score(y_test, y_pred))
'''  ##choosing the best K as the one that gives the highest testing score##
bestK_val = test_score.index(max(test_score))+3
print("best k : ", bestK_val)
KNNclassifier = KNeighborsClassifier(n_neighbors=bestK_val, p=5)
KNNclassifier.fit(x_train, y_train)
y_pred = KNNclassifier.predict(x_test)
evaluate(y_pred, KNNclassifier)
results.append(accuracy_score(y_test, y_pred))

#OR  

##choosing the best K as the one that gives testing score higher than the training score##
bestK_val=3
for i in range (len(test_score)):
    if test_score[i] > train_score[i]:
        bestK_val= i+3
        break
print("best k : ", bestK_val)
KNNclassifier = KNeighborsClassifier(n_neighbors=bestK_val, p=5)
KNNclassifier.fit(x_train, y_train)
y_pred = KNNclassifier.predict(x_test)
evaluate(y_pred, KNNclassifier)
results.append(accuracy_score(y_test, y_pred))
'''

print('\n\n####################### SVM ###########################')
SVMclassifier = SVC(kernel='linear', random_state = 1)
SVMclassifier.fit(x_train,y_train)
y_pred = SVMclassifier.predict(x_test)
evaluate(y_pred, SVMclassifier)
results.append(accuracy_score(y_test, y_pred))

print('\n###################### NAIVE BAYES ########################')
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
evaluate(y_pred, gnb)
results.append(accuracy_score(y_test, y_pred))

print('\n\n################# DECISION TREE ######################')
# 4) Create Decision Tree
clf = DecisionTreeClassifier(min_samples_leaf=50)
clf = clf.fit(x_train, y_train)
#print(plot_tree((clf)))
# 5) Predict from classifier
res =clf.predict(x_test)
#print(res)
# 6) Calculate accuracy
evaluate(y_predicted, clf)
results.append(accuracy_score(y_test, y_predicted))
tree.plot_tree(clf)
#=================================================================================
#-------------------------- Plotting Accuracy Comparison -------------------------
#=================================================================================
Results_DF = pd.DataFrame({'Classifiers': list(names), 'Accuracy': list(results)}).sort_values('Accuracy', ascending=True)
Results_DF.plot(x='Classifiers', y='Accuracy', kind='line')
plt.suptitle('Accuracy Comparison')
plt.show()