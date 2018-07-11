#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from itertools import product
from sklearn.ensemble import BaggingRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_log_error
import re 

import matplotlib.pyplot as plt
import seaborn as sns



def hp():

	houses=pd.read_csv("./input/train_scaled.csv")
	#split training and CV data
	houses_train, houses_test =  train_test_split(houses, test_size = 0.2)
	print(houses_train.head())
	plt.tight_layout()
	clf1 = DecisionTreeRegressor(max_depth=7)
	clf2 = KNeighborsRegressor(n_neighbors=9)
	clf3 = SVR(kernel='rbf')
	# clf3 = SVC(probability=True)
	clf4 = RandomForestRegressor(n_estimators=16)
	clf5 = LinearRegression()
	clf6 = Ridge()
	clf7 = Lasso()
	eclf = BaggingRegressor(base_estimator=clf3, n_estimators=5)
	
	train_x = houses_train.drop('SalePrice',axis=1)
	train_y = houses_train['SalePrice']

	test_x = houses_test.drop('SalePrice',axis=1)
	test_y = houses_test['SalePrice']

	nullcheck(train_x)
	# nullcheck(train_y)
	
	clf1 = clf1.fit(train_x,train_y)
	clf2 = clf2.fit(train_x,train_y)
	clf3 = clf3.fit(train_x,train_y)
	clf4 = clf4.fit(train_x,train_y)
	clf5 = clf5.fit(train_x,train_y)
	eclf = eclf.fit(train_x,train_y)


	y_pred = clf1.predict(train_x)
	print("Train Decision tree cassified accuracy: " + 	str(np.sqrt(mean_squared_log_error(y_pred, train_y))))
	y_pred = clf1.predict(test_x)
	print("Test Decision tree cassified accuracy: " + 	str(np.sqrt(mean_squared_log_error(y_pred, test_y))))

	ly= []
	lyy= []
	lx = []
	for i in range(0,1):
		clf1.set_params(max_depth=2+i)
		clf1 = clf1.fit(train_x,train_y)
		y_pred = clf1.predict(train_x)
		print("Train Decision tree cassified accuracy depth: " + str(3+i) + " : " + 	str(np.sqrt(mean_squared_log_error(y_pred, train_y))))
		ly.append(np.sqrt(mean_squared_log_error(y_pred, train_y)))
		y_pred = clf1.predict(test_x)
		print("Test Decision tree cassified accuracy depth: " + str(3+i) + " : " + 	str(np.sqrt(mean_squared_log_error(y_pred, test_y))))
		lx.append(3+i)
		lyy.append(np.sqrt(mean_squared_log_error(y_pred, test_y)))

	plt.plot(lx,lyy)
	plt.plot(lx,ly)
	plt.yscale('log')
	plt.title("Decision Tree")
	plt.savefig( './crossvali/d_tree.pdf')
	plt.clf()


	y_pred = clf2.predict(train_x)
	print("Train KNN7  cassified accuracy: " + 	str(np.sqrt(mean_squared_log_error(y_pred, train_y))))
	y_pred = clf2.predict(test_x)
	print("Test KNN7  cassified accuracy: " + 	str(np.sqrt(mean_squared_log_error(y_pred, test_y))))

	y_pred = clf3.predict(train_x)
	print("Train SVM cassified accuracy: " + 	str(np.sqrt(mean_squared_log_error(y_pred, train_y))))
	y_pred = clf3.predict(test_x)
	print("Test SVM cassified accuracy: " + 	str(np.sqrt(mean_squared_log_error(y_pred, test_y))))

	y_pred = clf4.predict(train_x)
	print("Train Random forest cassified accuracy: " + 	str(np.sqrt(mean_squared_log_error(y_pred, train_y))))
	y_pred = clf4.predict(test_x)
	print("Test Random forest cassified accuracy: " + 	str(np.sqrt(mean_squared_log_error(y_pred, test_y))))

	ly= []
	lyy = []
	lx = []
	for i in range(0,1):
		clf4.set_params(n_estimators=(3+i*3))
		clf4 = clf4.fit(train_x,train_y)
		y_pred = clf4.predict(train_x)
		print("Train Random Forest cassified accuracy depth: " + str(3+i*3) + " : " + 	str(np.sqrt(mean_squared_log_error(y_pred, train_y))))
		ly.append(np.sqrt(mean_squared_log_error(y_pred, train_y)))
		y_pred = clf4.predict(test_x)
		print("Test Random forest cassified accuracy depth: " + str(3+i*3) + " : " + 	str(np.sqrt(mean_squared_log_error(y_pred, test_y))))
		lx.append(3+i*3)
		lyy.append(np.sqrt(mean_squared_log_error(y_pred, test_y)))
	plt.plot(lx,ly)
	plt.plot(lx,lyy)
	plt.yscale('log')
	plt.title("Random Forest")
	plt.savefig( './crossvali/random_forest.pdf')
	plt.clf()


	y_pred = clf5.predict(train_x)
	print("Train linear cassified accuracy: " + 	str(np.sqrt(mean_squared_log_error(y_pred, train_y))))
	plt.scatter(train_y,train_y-y_pred)
	y_pred = clf5.predict(test_x)
	print("Test linear cassified accuracy: " + 	str(np.sqrt(mean_squared_log_error(y_pred, test_y))))
	plt.scatter(test_y,test_y-y_pred)
	res = test_y-y_pred
	print(res[np.all([res < -1], axis=0)])
	plt.title("Linear Residuals")
	plt.savefig( './crossvali/residuals.pdf')
	plt.clf()

	y_pred = eclf.predict(train_x)
	print("Train Voting (all 3) cassified accuracy: " + 	str(np.sqrt(mean_squared_log_error(y_pred, train_y))))
	y_pred = eclf.predict(test_x)
	print("Test Voting (all 3) cassified accuracy: " + 	str(np.sqrt(mean_squared_log_error(y_pred, test_y))))

	for name, clff in dict(zip(['ridge','lasso'], [clf6,clf7])).items() :
		ly= []
		lx = []
		lyy = []
		for i in range(0,2):
			clff.set_params(alpha=(0.00001 + i*0.2))
			clff = clff.fit(train_x,train_y)
			y_pred = clff.predict(train_x)
			print("Train " +str(name) + " alpha: " + str(0.00001 +i*0.2)+ " cassified accuracy: " + 	str(np.sqrt(mean_squared_log_error(y_pred, train_y))))
			ly.append(np.sqrt(mean_squared_log_error(y_pred, train_y)))
			y_pred = clff.predict(test_x)
			print("Test " +str(name) + " alpha: " + str(0.00001 +i*0.2) +  " cassified accuracy: " + 	str(np.sqrt(mean_squared_log_error(y_pred, test_y))))
			lyy.append(np.sqrt(mean_squared_log_error(y_pred, test_y)))
			lx.append(0.00001 +i*0.2)
		plt.plot(lx,ly)
		plt.plot(lx,lyy)
		plt.yscale('log')
		plt.title(name)
		plt.savefig( './crossvali/' + str(name) +'.pdf')
		plt.clf()

def nullcheck(df):
	null_columns=df.columns[df.isnull().any()]
	print(df[null_columns].isnull().sum())

def scale_inputs() :

	sns.set(style="whitegrid", color_codes=True)
	sns.set(font_scale=1)

	houses_train=pd.read_csv("./input/train.csv")
	print(houses_train[['MasVnrType','MasVnrArea']].head(30))

	houses_test = pd.read_csv("./input/test.csv")

	houses_train_length = houses_train.count
	houses_test_length = houses_test.count

	houses = pd.concat([houses_train,houses_test],keys=['train','test'])

	#convert text object to numeric
	houses['TotalBsmtSF'] = houses['TotalBsmtSF'].convert_objects(convert_numeric=True)

	#remove outliers in sale price
	upperlimit = np.percentile(houses_train.SalePrice.values, 99.5)
	print(upperlimit)
	houses_train.loc[houses_train['SalePrice']>upperlimit,'SalePrice'] = upperlimit

	plt.scatter(range(houses_train.shape[0]), houses_train["SalePrice"].values,color='orange')
	plt.title("Distribution of Sale Price")
	plt.xlabel("Number of Occurences")
	plt.ylabel("Sale Price");
	# plt.show()


	correlations=houses.corr()
	attrs = correlations.iloc[:-1,:-1] 
	threshold = 0.4
	important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0]).unstack().dropna().to_dict()

	unique_important_corrs = pd.DataFrame(list(set([(tuple(sorted(key)), important_corrs[key]) for key in important_corrs])), columns=['Attribute Pair', 'Correlation'])

	unique_important_corrs = unique_important_corrs.loc[
	    abs(unique_important_corrs['Correlation']).argsort()[::-1]]

	# print(unique_important_corrs)




	#lets see if there are any columns with missing values 
	null_columns=houses.columns[houses.isnull().any()]
	print(houses[null_columns].isnull().sum())


	# #Check correlation to fill missing
	houses['SqrtLotArea']=np.power(houses['LotArea'],0.001)
	print(houses[['LotFrontage','SqrtLotArea']].head(30))
	lot_front_area = houses['LotFrontage'].corr(houses['SqrtLotArea'])
	print(lot_front_area)
	houses = houses.drop(columns=['LotArea'])

	upperlimit = 180
	print(upperlimit)
	houses.loc[houses['LotFrontage']>upperlimit,'LotFrontage'] = upperlimit

	houses = normalise(houses,"LotFrontage")
	houses = normalise(houses,"SqrtLotArea")
	
	sns.jointplot(houses['LotFrontage'],houses['SqrtLotArea'],color='gold')
	# plt.show()
	houses['LotFrontage'] = houses['LotFrontage'].fillna(houses['SqrtLotArea'])

	print(houses[['LotFrontage','SqrtLotArea']].head(30))

	# #We can replace missing values with most frequent ones.
	houses["Electrical"] = houses["Electrical"].fillna('SBrkr')
	houses["Alley"] = houses["Alley"].fillna('None')




	# upperlimit = np.percentile(houses.TotalBsmtSF.values, 99.5)
	# houses.loc[houses['TotalBsmtSF']>upperlimit,'TotalBsmtSF'] = upperlimit

	# plt.scatter(houses.TotalBsmtSF, houses["SalePrice"].values,color='orange')
	# plt.title("TotalBsmtSF Vs SalePrice ")
	# plt.ylabel("SalePrice")
	# plt.xlabel("Total Basement in sq feet")
	plt.show()

	basement_cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','BsmtCond']
	# print(houses[basement_cols].head())
	# print(houses[basement_cols][houses['BsmtQual'].isnull()==True])
	# print(houses[basement_cols].head())


	for col in basement_cols:
		if houses[col].dtype==np.object:
			houses[col] = houses[col].fillna('None')
		else:
			houses[col] = houses[col].fillna(0)

	#If fireplace quality is missing that means that house doesn't have a fireplace
	houses["FireplaceQu"] = houses["FireplaceQu"].fillna('None')
	print(pd.crosstab(houses.Fireplaces, houses.FireplaceQu))

	garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']
	# print(houses[garage_cols][houses['GarageType'].isnull()==True])


	for col in garage_cols:
	    if houses[col].dtype==np.object:
	        houses[col] = houses[col].fillna('None')
	    else:
	        houses[col] = houses[col].fillna(0)

	houses["PoolQC"] = houses["PoolQC"].fillna('None')
	houses["Fence"] = houses["Fence"].fillna('None')
	houses["MiscFeature"] = houses["MiscFeature"].fillna('None')

	null_columns=houses.columns[houses.isnull().any()]
	print(houses[null_columns].isnull().sum())


	# print(houses[['MasVnrType','MasVnrArea']].groupby(['MasVnrType'], as_index=False).agg(np.size).sort_values(by='MasVnrType', ascending=False))
	
	sns.countplot(houses['MasVnrType'])
	plt.xlabel("MasVnrType")
	# plt.show()

	# houses['MasVnrArea'].plot(kind='hist')
	# plt.xlabel("MasVnrArea")
	# print(houses.loc[houses['MasVnrType'] == houses['MasVnrType'].mode()])
	houses.loc[(houses['MasVnrArea'].notnull()) & (houses['MasVnrType'].isnull()),'MasVnrType'] = houses['MasVnrType'].mode().iloc[0]
	# houses['MasVnrType'] = houses['MasVnrType'].fillna('None')
	print("MasVnrArea")
	print(houses[null_columns][houses['MasVnrType'].isnull()])

	houses['MasVnrArea'] = houses['MasVnrArea'].fillna(0)
	houses['MasVnrType'] = houses['MasVnrType'].fillna('None')

	# plt.show()
	# print(houses[["MasVnrType", "GarageType"]].groupby(['MasVnrType'], as_index=False).agg(np.size).sort_values(by='GarageType', ascending=False))

	null_columns=houses.columns[houses.isnull().any()]
	for col in null_columns:
		houses[col]=houses[col].fillna(value=houses[col].mode().iloc[0])
		print("\n\n\n\n\n" + str(col) + "\n")
		print(houses[houses.columns][houses[col].isnull()])

	null_columns=houses.columns[houses.isnull().any()]
	print(houses[null_columns].isnull().sum())




	#Got columns with outliers from above, removing anomalous data	
	
	
	adjust_cols = ['1stFlrSF','BsmtFinSF1','BsmtFinSF2','EnclosedPorch','GarageYrBlt','MiscVal','OpenPorchSF','TotalBsmtSF']

	for col in adjust_cols:
		maximum = np.percentile(houses[col][houses[col]!=0].values, 99.5)*1.4
		minimum = np.percentile(houses[col][houses[col]!=0].values,0.5)*0.6
		houses.loc[(houses[col]>maximum) & (houses[col]!=0),col] = maximum/1.4
		houses.loc[(houses[col]<minimum) & (houses[col]!=0),col] = minimum/0.6
		print(str(col) + "  max: " + str(maximum) +  " min: " + str(minimum))


	#Removing highly correlated features by taking combinations and ratios. 
	houses['GarageArea_Cars'] = houses['GarageCars'] / houses['GarageArea']
	houses = houses.drop(columns=['GarageCars'])
	houses['GarageArea_Cars'] = houses['GarageArea_Cars'].fillna(0)
	# houses['GrLivArea_TotRmsAbvGrd'] = houses['GrLivArea']*houses['TotRmsAbvGrd']
	# houses = houses.drop(columns=['TotRmsAbvGrd'])
	houses['1stFlrSF_TotalBsmtSF'] = houses['1stFlrSF'] - houses['TotalBsmtSF']
	houses = houses.drop(columns=['TotalBsmtSF'])
	houses['BedroomAbvGr_TotRmsAbvGrd'] = houses['BedroomAbvGr'] - houses['TotRmsAbvGrd']
	houses = houses.drop(columns=['TotRmsAbvGrd'])
	houses['2ndFlrSF_GrLivArea'] = houses['2ndFlrSF'] - houses['GrLivArea']
	houses = houses.drop(columns=['GrLivArea'])

	correlations=houses.corr()
	attrs = correlations.iloc[:-1,:-1] # all except target

	threshold = 0.5
	important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0]).unstack().dropna().to_dict()

	unique_important_corrs = pd.DataFrame(list(set([(tuple(sorted(key)), important_corrs[key]) for key in important_corrs])), columns=['Attribute Pair', 'Correlation'])

	    # sorted by absolute value
	unique_important_corrs = unique_important_corrs.loc[
	    abs(unique_important_corrs['Correlation']).argsort()[::-1]]

	print(unique_important_corrs)


	attrs = attrs['SalePrice']
	#Take absolute correlations, sort values decending, get index of columns from 1 to 7
	important_columns = np.abs(attrs).sort_values(ascending=False).index[1:7]
	for i in range(0,len(important_columns)):
		for j in range(i,len(important_columns)):
			houses[str(important_columns[i]+ "_" +important_columns[j])] = houses[important_columns[i]]*houses[important_columns[j]]


	for column in houses :
		# print(column)
		if pd.api.types.is_object_dtype(houses[column].dtype):
			# print("here instead")
			# print(houses[column].head())
			sns.countplot(houses[column])

		else:
			# print("here!")
			houses = normalise(houses,column)
			houses[column].plot(kind='hist')

		plt.yscale('log')
		plt.xlabel(column)
		plt.savefig( './outputfigs/' + str(column) +'.pdf')
		plt.clf()

	for column in houses:
		if pd.api.types.is_object_dtype(houses[column].dtype):
			houses = split_discrete_features(houses,column)



	houses_train=houses.loc['train',:]
	houses_test=houses.loc['test',:]
	houses_train.to_csv('./input/train_scaled.csv')
	houses_test.to_csv('./input/test_scaled.csv')

	print(houses.head())
	print(houses.info())


def normalise(df,feature_name):
	#Normalise feature between 0 and 1
    result = df.copy()

   	# max_value = df[feature_name].max()
    # min_value = df[feature_name].min()

    max_value = df[feature_name].max()
    min_value = df[feature_name].min()

    result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    
    return result

def split_discrete_features(df,feature_name):
	#Split features with discrete values into seperate features
	unique_list = df[feature_name].unique() 
	for i in unique_list :
		new_df_name = str(feature_name) + "_" + str(i)
		df[new_df_name] = df[feature_name] == i
	df = df.drop(feature_name,axis=1)
	return df 

# scale_inputs()
hp()
