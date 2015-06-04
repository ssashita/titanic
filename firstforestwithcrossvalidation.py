""" Writing my first randomforest code.
Author : AstroDave
Date : 23rd September 2012
Revised: 15 April 2014
please see packages.python.org/milk/randomforests.html for more

""" 
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

TIMES=5

def validation_error(arr1,arr2):
    return sum([abs(arr1[i]-arr2[i]) for i in range(len(arr1))])
    
# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('train.csv', header=0)        # Load the train file into a dataframe

# I need to convert all strings to integer classifiers.
# I need to fill in the missing values of the data and make it complete.

# female = 0, Male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

# All missing Embarked -> just make them embark from most common place
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

# All the ages with no data -> make the median of all Ages
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 


# TEST DATA
test_df = pd.read_csv('test.csv', header=0)        # Load the test file into a dataframe

# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# All missing Embarked -> just make them embark from most common place
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
# Again convert all Embarked strings to int
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)


# All the ages with no data -> make the median of all Ages
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 


# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values


print 'Training...'
errorlist=[]
validation_data_arr = np.array([train_data[4*i].tolist() for i in range(len(train_data)/4) ])
valid_check_subscript=[4*i for i in range(len(train_data)/4) ]
small_check_subscript=[4*i+j for i in  range(len(train_data)/4) for j in [1,2,3] if j+4*i < len(train_data)]
print [i for i in valid_check_subscript if i in small_check_subscript]
#small_train_data = np.array([train_data[i].tolist() for train_data[i] not in validation_data_arr])
small_train_data = np.array([train_data[4*i+j].tolist() for i in  range(len(train_data)/4) for j in [1,2,3] if j+4*i < len(train_data)])

for times in range(TIMES):
    for paramsplit in range(100)[2:]:
        forest = RandomForestClassifier(n_estimators=100, min_samples_split=paramsplit, max_features=None)
#        print 'Size of train data  array is ', small_train_data.size
        forest = forest.fit( small_train_data[0::,1::], small_train_data[0::,0] )

        valid_out = forest.predict(validation_data_arr[0:,1:]).astype(int)
        error= validation_error(validation_data_arr[0:,0],valid_out)
        print 'Param split value is ', paramsplit, 'Error  is ', error
        errorlist.append(error)
        print '---------------------------------------'

error_arr = np.array(errorlist,int)
error_arr = error_arr.reshape((TIMES,98))
error_arr=error_arr.transpose()
error_final_list= [np.mean(error_arr[i]) for i in range(error_arr.shape[0])]
std_final_list = [np.std(error_arr[i]) for i in range(error_arr.shape[0])]
def keyfunc(item):
    return item[1]
print 'Errors for different split param vals'
print len(error_final_list)
#print 'Std devs for different split param vals'
#print std_final_list
decision_lst= zip(range(100)[2:],error_final_list)
decision_lst = sorted(decision_lst,key=keyfunc)
print '(split pram,Minimum error) is ', decision_lst[0]
print [x for x in decision_lst if x==decision_lst[0] ]
