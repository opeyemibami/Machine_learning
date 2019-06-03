# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder , OneHotEncoder

    
traindemographics = pd.read_csv('traindemographics.csv')
trainperf = pd.read_csv('trainperf.csv')
trainprevloans = pd.read_csv('trainprevloans.csv')


testdemographics = pd.read_csv('testdemographics.csv')
testperf = pd.read_csv('testperf.csv')
testprevloans = pd.read_csv('testprevloans.csv')

merge_train = pd.merge(trainperf,traindemographics, on = 'customerid', how = 'left')
merge_test = pd.merge(testperf,testdemographics, on = 'customerid', how = 'left')


sample_summit = pd.read_csv('SampleSubmission.csv')
drop = ['systemloanid','approveddate','creationdate','referredby','birthdate','longitude_gps','latitude_gps','bank_name_clients', 'bank_branch_clients','level_of_education_clients','employment_status_clients']
for colm in drop:
    merge_train = merge_train.drop(colm, axis = 1)
    merge_test = merge_test.drop(colm, axis = 1)

bank_account_type  = {'Other': 0, 'Savings': 1, 'Current': 2}
merge_train['bank_account_type'] =  merge_train['bank_account_type'].map(bank_account_type)
merge_test['bank_account_type'] =  merge_test['bank_account_type'].map(bank_account_type)

employment_status_clients = {'Permanent': 0, 'Unemployed': 1, 'Self-Employed': 2, 'Student' : 3, 'Retired' : 4, 'Contract' : 5 }
merge_train['employment_status_clients'] =  merge_train['employment_status_clients'].map(employment_status_clients)
merge_test['employment_status_clients'] =  merge_test['employment_status_clients'].map(employment_status_clients)


#encodingList = ['good_bad_flag','bank_account_type','employment_status_clients']
#for columns in encodingList:
#    labelEncoder_merge_train = LabelEncoder() 
#    merge_train[columns] = labelEncoder_merge_train.fit_transform(merge_train[columns])
#    
#    

X_df = merge_train.drop('good_bad_flag',axis = 1)
y_df = merge_train['good_bad_flag']
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y_df)

Xtesting = merge_test.values
X = X_df.values
y = y_df.values

imputercolumns = [5]
for clm in imputercolumns:
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values = 'NaN',strategy = 'most_frequent',axis=0)
    X[:,clm:clm+1]=imputer.fit_transform(X[:,clm:clm+1])
    Xtesting[:,clm:clm+1]=imputer.fit_transform(Xtesting[:,clm:clm+1])
 
    
X_df = pd.DataFrame(X)
Xtesting =pd.DataFrame(Xtesting)


train = X_df.drop([0],axis = 1).values
test = Xtesting.drop([0],axis = 1).values


onehot = OneHotEncoder(categorical_features=[4])
train= onehot.fit_transform(train).toarray()
test = onehot.fit_transform(test).toarray()

drop = [0]
for items in drop:
    train = (pd.DataFrame(train)).drop(items,axis = 1)
    test = (pd.DataFrame(test)).drop(items,axis = 1)

train = train.values    
test = test.values

from sklearn.preprocessing import StandardScaler
sc_train = StandardScaler()
sc_test = StandardScaler()
train = sc_train.fit_transform(train)
test = sc_test.fit_transform(test)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.3, random_state=42)
  
from sklearn.ensemble import RandomForestClassifier
RFClassifier = RandomForestClassifier(n_estimators = 50)
RFClassifier.fit(train,y) 
y_pred = RFClassifier.predict(train)
y_predt = RFClassifier.predict(test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from xgboost import XGBClassifier
xgbClassifier = XGBClassifier(n_estimators=500 ,learning_rate=0.1,random_state=42)
xgbClassifier.fit(train,y) 
y_predXgb = xgbClassifier.predict(train)
y_predXgbt = xgbClassifier.predict(test)

cm = confusion_matrix(y_test, y_predXgb)

from sklearn.neighbors import KNeighborsClassifier
KNclassifier = KNeighborsClassifier(n_neighbors = 20,metric = 'minkowski',p =2)
KNclassifier.fit(train,y)
y_predKN = KNclassifier.predict(train)
y_predKNt = KNclassifier.predict(test)

cm = confusion_matrix(y_test, y_predKN)

from sklearn.ensemble import GradientBoostingClassifier
gdr = GradientBoostingClassifier(n_estimators = 1000,random_state=42, learning_rate = 0.02, max_depth = 2)
gdr.fit(train,y)
y_predgrd = gdr.predict(train)
y_predgrdt = gdr.predict(test)

cm = confusion_matrix(y_test, y_predgrd)

stacked_prediction1 = np.column_stack((y_pred,y_predXgb,y_predKN,y_predgrd))
stacked_predictionTest1 = np.column_stack((y_predt,y_predXgbt,y_predKNt,y_predgrdt))


xgb_metal_Classifier = XGBClassifier(n_estimators=10 ,learning_rate=0.1,random_state=42)
xgb_metal_Classifier.fit(stacked_prediction1,y)
meta_prediction1 = xgb_metal_Classifier.predict(stacked_predictionTest1)












