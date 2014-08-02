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

# load model
output_model_file = 'C:/LearningMaterials/Kaggle/Mlsp/mode2.pkl'
net = pickle.load( open( output_model_file, 'rb' ))

#Load the Test Set as X_Test
df_4 = pd.read_csv('C:/LearningMaterials/Kaggle/Mlsp/test_FNC.csv')
df_5 = pd.read_csv('C:/LearningMaterials/Kaggle/Mlsp/test_SBM.csv')
df_test_fnc = df_4.ix[:, 1:]
df_test_sbm = df_5.ix[:,1:]
np_test_fnc = df_test_fnc.values
np_test_sbm = df_test_sbm.values
X_test = np.hstack((np_test_fnc,np_test_sbm))
y_test_dummy = np.zeros((119748,1))

print "Dimensions of test X"
print X_test.shape
in_size = X_test.shape[1]
ds_test = CDS( in_size, class_labels=['Healthy','Schizo'] )
ds_test.setField( 'input', X_test )
ds_test.setField( 'target', y_test_dummy)

p = net.activateOnDataset( ds_test )
#for pred in p:
#    print pred[0]
np.savetxt('submission.csv', p, delimiter=",", fmt = '%1.4f')

#X_test_identifers = df_1.ix[:,0].values
#y_predict = clf.predict_proba(X_test)

#print y_predict
#print y
#y_predict = y_predict[:,1]
#np.savetxt('submission.csv', y_predict, delimiter=",", fmt = '%1.4f')

print "done."

#final_predictions = np.hstack((X_test_identifers, y_predict))

#print y_predict

#np.savetxt('submission.csv', final_predictions, delimiter=",", fmt = '%1.4f')
