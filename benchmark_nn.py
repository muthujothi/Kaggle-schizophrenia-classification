import pandas as pd
import numpy as np
from sklearn import linear_model
import cPickle as pickle
from math import sqrt
#from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.datasets import ClassificationDataSet as CDS
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure.modules   import SigmoidLayer

hidden_size = 100
output_model_file = 'C:/LearningMaterials/Kaggle/Mlsp/mode2.pkl'

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
y = y.reshape(-1, 1)
print "Dimensions of input feature vector X "
print X.shape
input_size = X.shape[1]


#Get a linear model from the sklearn
#clf = linear_model.LogisticRegression(C=0.16,penalty='l1', tol=0.001, fit_intercept=True)
#clf.fit(X, y)

#Get a network that trains based on backpropagation method using the training data
ds = CDS( input_size, class_labels=['Healthy','Schizo'] )
ds.setField( 'input', X )
ds.setField( 'target', y)

print len(ds)

# init and train
net = buildNetwork( input_size, hidden_size, 1, bias = True, outclass=SigmoidLayer )#feature vector, hidden layer size, 
trainer = BackpropTrainer( net,ds )
trainer.trainUntilConvergence( verbose = True, validationProportion = 0.15, maxEpochs = 1000, continueEpochs = 10 )

pickle.dump( net, open( output_model_file, 'wb' ))

p = net.activateOnDataset( ds )
np.savetxt('nn_sub2.csv', p, delimiter=",", fmt = '%1.4f')

#print net['in']
#print net['hidden0']
#print net['out']


#for pred in p:
    #print round(pred[0],4)
#    print pred[0]




