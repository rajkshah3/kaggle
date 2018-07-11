#!/usr/bin/env python3

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from tf import nn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
pylab.show()
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import VotingClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import re 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

def titanic() :
	from subprocess import check_output

	test_read = 100000

	train = pd.read_csv('../data/train.csv', header=0,nrows=test_read)
	test = pd.read_csv('../data/test.csv', header=0,nrows=test_read)

	data = [train , test]
	# train = pd.concat(data)


	train = prepare_inputs(train)

	# print(train.head(n=20))

#
# Do machine learning on training data
#
	# do_machine_learning(train)
	
#### Do Tensorflow

	do_tensor_flow(train)

#Print stuff out
	# # fig, ax = plt.subplots()


	# fig = pylab.figure(1)
	# fig.suptitle('Sex && Survived', fontsize=14, fontweight='bold')

	# dummy = train[train['Survived']==1]
	# dummy['Sex'].plot(kind='hist')
	# # train[train['Survived']==1].ix['Sex'].plot(kind='hist')

	# # pylab.show()


	# fig = pylab.figure(2)

	# fig.suptitle('Sex && died', fontsize=14, fontweight='bold')
	# # dummy = train[train['Survived']==1]
	# # dummy['Sex'].plot(kind='hist')

	# train.loc[train['Survived']==0,"Sex"].plot(kind='hist')
	# # pylab.show()

	# fig = pylab.figure(3)
	# fig.suptitle('Fare', fontsize=14, fontweight='bold')


	# train['Fare'].plot(kind='hist')
	# # pylab.show()
	# fig = pylab.figure(4)

	# fig.suptitle('Age', fontsize=14, fontweight='bold')

	# train['Age'].plot(kind='hist')
	# pylab.show()

	# fig = pylab.figure(5)

	# fig.suptitle('Sex', fontsize=14, fontweight='bold')

	# train['Age'].plot(kind='hist')
	# pylab.show()
	input("Press Enter to continue...")


def data_clean(data): #Group family and drop unused identifiers
    data['Family'] = data['Parch'] + data['SibSp']
    data = data.drop(['Parch','SibSp','Ticket','PassengerId'],axis=1)
    return data


def normalize(df,feature_name):
    result = df.copy()

   	# max_value = df[feature_name].max()
    # min_value = df[feature_name].min()

    max_value = df[feature_name].max()
    min_value = df[feature_name].min()

    result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    
    return result

def fix_string_inputs(df) :
	# print(df[['Name','Age']].head(n=10))
	df['Maiden'] = df['Name'].apply(maiden_name)
	df['Name'] = df['Name'].apply(remove_brackets)
	# print(df[['Name','Maiden','Age']].head(n=10))
	df['Title'] = df['Name'].apply(get_title)
	# print(df[['Name','Maiden','Title','Age']].head(n=10))
	df['Name'] = df['Name'].apply(remove_title)
	# print(df[['Name','Maiden','Title','Age']].head(n=10))
	df['Num_names'] = df['Name'].apply(num_words)
	df['Num_maiden_names'] 	= df['Maiden'].apply(num_words)
	df['Name_length']		= df['Name'].apply(len)
	df['Name_maiden_length']= df['Maiden'].apply(len)
	df['Cabin'] = df['Cabin'].fillna("Z")
	df['Cabin'] = df['Cabin'].apply(get_first_char)
	df = df.drop(['Name','Maiden'],axis=1)
	df['Title'] = df['Title'].apply(group_title)

	return df

def prepare_inputs(df) :

		# print(train)
	df = data_clean(df)
	# print(train)
	print("Train Before norm")
	# print(train['Age'].describe())

	print(df.columns)

	# print("Train mean: " + str(train['Age'].mean()))

	# print("Train After norm")

	df['Pclass'] = df['Pclass'].fillna(df['Pclass'].mode())
	df = fix_string_inputs(df)


#Set age based on average of people with the same title
	mean_ages = df.groupby(['Title'])['Age'].mean()
	df = df.set_index(['Title'])
	df['Age'] = df['Age'].fillna(mean_ages)
	df = df.reset_index()

	# df['Age'] = df['Age'].fillna(df['Age'].mean())


	df =	split_discrete_features(df,"Title")
	df = 	split_discrete_features(df,"Sex")
	df = 	split_discrete_features(df,"Pclass")
	df = 	split_discrete_features(df,"Embarked")
	df = 	split_discrete_features(df,"Cabin")

	df = normalize(df,"Age")

	return df

def remove_brackets(input_string) :

	output = re.sub("[\(\[].*?[\)\]]", "", input_string)
	return output

def maiden_name(input_string) :
	# print(input_string)
	title_search = re.search('\(([^)]+)', input_string)
	if(title_search) :
		# print(title_search.group(1))
		return title_search.group(1)
	else :
		return ""

def num_words(input):
	return len(input.split())

def get_first_char(input):
	mystring = input
	cabin = mystring[0]
	return cabin

def get_title(input):
	result = input[input.find(", ")+1:input.find(".")]
	if(result) :
		return result
	else : 
		return ""

def group_title(input):
	search_dict = {" Mr": "Mr"," Mrs" : "Mrs"," Master": "Master"," Miss" : "Miss"," Mlle" : "Miss", " Ms" : "Miss", " Lady": "Royal", " Sir" : "Royal", " the Countess" : "Royal", " Mme" : "Miss"  }
	if input in search_dict :
		return search_dict[input]
	else :
		return "Other"

def remove_title(input):
	result = input[input.find(", ")+1:input.find(".")]
	output = input.replace(result + ".",'')
	return output

def split_discrete_features(df,feature_name):
	unique_list = df[feature_name].unique() 
	for i in unique_list :
		new_df_name = str(feature_name) + "_" + str(i)
		df[new_df_name] = df[feature_name] == i
	df = df.drop(feature_name,axis=1)
	return df 

def do_machine_learning(train) :
	train, test =  train_test_split(train, test_size = 0.1)
	train_x = train.drop('Survived',axis=1)
	train_y = train['Survived']
	test_x = test.drop('Survived',axis=1)
	test_y = test['Survived']


	clf1 = DecisionTreeClassifier(max_depth=6)
	clf2 = KNeighborsClassifier(n_neighbors=15)
	clf3 = SVC(kernel='rbf', probability=True)
	# clf3 = SVC(probability=True)
	clf4 = RandomForestClassifier(n_estimators=50)
	eclf = VotingClassifier(estimators=[('dt', clf1), ('svc', clf3)], voting='soft', weights=[2,1])


	clf1 = clf1.fit(train_x,train_y)
	clf2 = clf2.fit(train_x,train_y)
	clf3 = clf3.fit(train_x,train_y)
	clf4 = clf4.fit(train_x,train_y)
	eclf = eclf.fit(train_x,train_y)


	y_pred = clf1.predict(train_x)
	print("Train Decision tree cassified accuracy: " + 	str(round(accuracy_score(y_pred, train_y) * 100, 2)))
	y_pred = clf1.predict(test_x)
	print("Test Decision tree cassified accuracy: " + 	str(round(accuracy_score(y_pred, test_y) * 100, 2)))

	y_pred = clf2.predict(train_x)
	print("Train KNN7  cassified accuracy: " + 	str(round(accuracy_score(y_pred, train_y) * 100, 2)))
	y_pred = clf2.predict(test_x)
	print("Test KNN7  cassified accuracy: " + 	str(round(accuracy_score(y_pred, test_y) * 100, 2)))

	y_pred = clf3.predict(train_x)
	print("Train SVM cassified accuracy: " + 	str(round(accuracy_score(y_pred, train_y) * 100, 2)))
	y_pred = clf3.predict(test_x)
	print("Test SVM cassified accuracy: " + 	str(round(accuracy_score(y_pred, test_y) * 100, 2)))

	y_pred = clf4.predict(train_x)
	print("Train Random forest cassified accuracy: " + 	str(round(accuracy_score(y_pred, train_y) * 100, 2)))
	y_pred = clf4.predict(test_x)
	print("Test Random forest cassified accuracy: " + 	str(round(accuracy_score(y_pred, test_y) * 100, 2)))

	y_pred = eclf.predict(train_x)
	print("Train Voting (all 3) cassified accuracy: " + 	str(round(accuracy_score(y_pred, train_y) * 100, 2)))
	y_pred = eclf.predict(test_x)
	print("Test Voting (all 3) cassified accuracy: " + 	str(round(accuracy_score(y_pred, test_y) * 100, 2)))


def print_stuff(train) :
	
	#Need to have dataset of two entries to group by 1. Survived entry contributes no information here


	print(train[["Gender", "Survived"]].groupby(['Gender'], as_index=False).agg(np.size).sort_values(by='Gender', ascending=False))

	print(train[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).agg(np.size).sort_values(by='Pclass', ascending=False))

	print(train[["Family", "Survived"]].groupby(['Family'], as_index=False).agg(np.size).sort_values(by='Family', ascending=False))

	print(train[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).agg(np.size).sort_values(by='Embarked', ascending=False))

	print(train[["Gender", "Survived"]].groupby(['Gender'], as_index=False).agg(np.size).sort_values(by='Gender', ascending=False))

	print(train[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).agg(np.size).sort_values(by='Pclass', ascending=False))

	print(train[["Family", "Survived"]].groupby(['Family'], as_index=False).agg(np.size).sort_values(by='Family', ascending=False))

	# print(train[['Gender','Pclass']].groupby('Gender').agg(np.size))
	fig = pylab.figure()

	train.loc[train['Survived']==0,"Gender"].value_counts().plot(kind='bar', color='red',position=1,label='survived')
	train.loc[train['Survived']==1,"Gender"].value_counts().plot(kind='bar', color='blue',position=0,label='died')
	pylab.legend(loc='upper left')

	pylab.show()

	fig = pylab.figure()

	train.loc[train['Survived']==0,"Pclass"].value_counts().plot(kind='bar', color='red',position=1,label='survived')
	train.loc[train['Survived']==1,"Pclass"].value_counts().plot(kind='bar', color='blue',position=0,label='died')
	pylab.legend(loc='upper left')

	pylab.show()


def plot_stuff(train) :

	fig = pylab.figure(1)
	fig.suptitle('Gender && Survived', fontsize=14, fontweight='bold')

	dummy = train[train['Survived']==1]
	dummy['Gender'].plot(kind='hist')
	# train[train['Survived']==1].ix['Gender'].plot(kind='hist')

	# pylab.show()


	fig = pylab.figure(2)

	fig.suptitle('Gender && died', fontsize=14, fontweight='bold')
	# dummy = train[train['Survived']==1]
	# dummy['Gender'].plot(kind='hist')

	train.loc[train['Survived']==0,"Gender"].plot(kind='hist')
	# pylab.show()

	fig = pylab.figure(3)
	fig.suptitle('Fare', fontsize=14, fontweight='bold')


	train['Fare'].plot(kind='hist')
	# pylab.show()
	fig = pylab.figure(4)

	fig.suptitle('Age', fontsize=14, fontweight='bold')

	train['Age'].plot(kind='hist')
	# pylab.show()

	fig = pylab.figure(5)

	fig.suptitle('Gender', fontsize=14, fontweight='bold')

	train['Gender'].plot(kind='hist')
	pylab.show()


def do_tensor_flow(train) :
	train, test =  train_test_split(train, test_size = 0.1)
	train_x = train.drop('Survived',axis=1)
	train_y = train['Survived']
	test_x = test.drop('Survived',axis=1)
	test_y = test['Survived']

	nn(train_x,train_y,test_x,test_y)

titanic()

