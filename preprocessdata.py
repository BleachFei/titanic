import pandas as pd
import numpy as np

# For .read_csv, always use header=0 when you know row 0 is the header row
train_df = pd.read_csv('train.csv', header = 0)
test_df = pd.read_csv('test.csv', header = 0)
df = pd.concat([train_df,test_df])

def preprocess():
# let's decide female = 0 and male = 1
    global df
    df['Gender'] = 4
    df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )
    df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    #build another reference table to calculate what each of these medians
    median_ages = np.zeros((2,3))
    # populating the array
    for i in range(0, 2):
    	for j in range(0, 3):
    		median_ages[i,j] = df[(df['Gender'] == i) & \
                  (df['Pclass'] == j+1)]['Age'].dropna().median()
    	 
   # Make a copy of Age
   # assigning it an appropriate value out of median_ages
    df['AgeFill'] = df['Age']
    for i in range(0, 2):
    	for j in range(0, 3):
    		df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
    				'AgeFill'] = median_ages[i,j]
    					
    # records whether the Age was originally missing
    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
    
    #replace missing values with mode
    df.Embarked[df.Embarked.isnull()] = df.Embarked.dropna().mode().values
    	
    #creat dummy varibles from raw data
    dummies_df = pd.get_dummies(df.Embarked)
    #remana the columns to Embarked_S...
    dummies_df = dummies_df.rename(columns=lambda x:'Embarked_'+str(x))
    df = pd.concat([df,dummies_df],axis=1)
    	
    # we could collect those together as a FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['Age*Class'] = df.AgeFill * df.Pclass
    
    #.dtypes to show only the columns which are 'object', 
    #which for pandas means it has strings
    #df.dtypes[df.dtypes.map(lambda x: x=='object')]
    


