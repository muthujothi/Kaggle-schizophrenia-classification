import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier # random forest
import cPickle as pickle

output_model_file = 'C:/LearningMaterials/Kaggle/Mlsp/rf_1.pkl'

#Load the FNC train data as a dataframe
df_1 = pd.read_csv('C:/LearningMaterials/Kaggle/Mlsp/train_FNC.csv')

#Load the SBM train data as a dataframe
df_2 = pd.read_csv('C:/LearningMaterials/Kaggle/Mlsp/train_SBM.csv')

#Leave the first column in both the datasets to load the features as a dataframe
df_train_fnc = df_1.ix[:, 1:]
df_train_sbm = df_2.ix[:,1:]

#Geta numpy (n_samples X n_features) representation for each of the data frame.
np_fnc = df_train_fnc.values
np_sbm = df_train_sbm.values

#column wise stack both the numpy matrices to get a feature matrix X
X = np.hstack((np_fnc,np_sbm))

#shud be 4 X 410
#print X.shape

#Load the labels data
df_3 = pd.read_csv('C:/LearningMaterials/Kaggle/Mlsp/train_labels.csv')
df_train_labels = df_3.ix[:,1]
y = df_train_labels.values
print "Dimensions of train X and Y"
print X.shape
print y.shape

#Get a linear model from the sklearn
#clf = linear_model.LogisticRegression(C=0.16,penalty='l1', tol=0.001, fit_intercept=True)
tuned_parameters = [{'max_features': ['sqrt'], 'n_estimators': [1000]}]
clf = GridSearchCV( RandomForestClassifier(min_samples_split=1), tuned_parameters, cv=3, verbose=2 ).fit(X, y)
#clf = linear_model.LogisticRegression(C=0.16,penalty='l2', tol=0.001, fit_intercept=True)
#clf.fit(X, y)
print 'Best parameters set found on development set:'
print clf.best_estimator_
y_predict_train = clf.predict_proba(X)
#print 'prediction accuracy: %.4f' % (1 - (1. / len(y) * sum( y_predict_train != y )))
pickle.dump( clf, open( output_model_file, 'wb' ))
print 'Model Saved and working on test set'
y_predict_train = y_predict_train[:,1]
np.savetxt('submission_train.csv', y_predict_train, delimiter=",", fmt = '%1.4f')

#Load the Test Set as X_Test
df_4 = pd.read_csv('C:/LearningMaterials/Kaggle/Mlsp/test_FNC.csv')
df_5 = pd.read_csv('C:/LearningMaterials/Kaggle/Mlsp/test_SBM.csv')
df_test_fnc = df_4.ix[:, 1:]
df_test_sbm = df_5.ix[:,1:]
np_test_fnc = df_test_fnc.values
np_test_sbm = df_test_sbm.values
X_test = np.hstack((np_test_fnc,np_test_sbm))

print "Dimensions of test X"
print X_test.shape
#X_test_identifers = df_1.ix[:,0].values
y_predict = clf.predict_proba(X_test)

#print y_predict
#print y
y_predict = y_predict[:,1]
np.savetxt('submission.csv', y_predict, delimiter=",", fmt = '%1.4f')

print "done."


#final_predictions = np.hstack((X_test_identifers, y_predict))

#print y_predict

#np.savetxt('submission.csv', final_predictions, delimiter=",", fmt = '%1.4f')
