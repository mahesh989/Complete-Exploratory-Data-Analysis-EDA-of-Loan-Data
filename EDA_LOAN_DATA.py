import numpy as np
import pandas as pd
import requests
import seaborn as sns

#readin data
df = pd.read_csv('/Users/mahesh/Library/CloudStorage/OneDrive-CharlesDarwinUniversity/Workspace//EDA/bank/train_indessa.csv', sep=',')


df.shape #gives number of rows and columns
df.info()  #provides datatype and respective data info null or not
df.head()  #gives first 5 rows by default
df.tail(10) #gives last 10 rows as we desire 


#CHECK FOR NULL VALUES WITH COUNT
null_cols = df.columns[df.isnull().any()]
null_df = df[null_cols].isnull().sum().to_frame(name='Null Count')\
          .merge(df[null_cols].isnull().mean().mul(100).to_frame(name='Null Percent'), left_index=True, right_index=True)
null_df_sorted = null_df.sort_values(by='Null Count', ascending=False)
print(null_df_sorted)


df[df.duplicated()].shape[0] # gives the number of duplicate entries


#sub data_frame
df_new = df[['funded_amnt_inv', 'term', 'int_rate', 'emp_length', 'emp_title', 'annual_inc',
             'verification_status', 'purpose', 'addr_state', 'dti', 'initial_list_status',
             'total_rec_int', 'total_rec_late_fee', 'application_type', 'tot_coll_amt',
             'tot_cur_bal', 'loan_status']].rename(columns={
    'funded_amnt_inv': 'BANK_INVESTMENT',
    'term': 'TERM',
    'int_rate': 'INTEREST_RATE',
    'emp_length': 'EMPLOYEMENT_DURATION',
    'emp_title': 'EMPPLOYMENT_TITLE',
    'annual_inc': 'ANNUAL_INCOME',
    'verification_status': 'STATUS',
    'purpose': 'LOAN_PURPOSE',
    'addr_state': 'STATE',
    'dti': 'DTI',
    'initial_list_status': 'INITIAL_LIST_STATUS',
    'total_rec_int': 'RECEIVED_INTEREST_TOTAL',
    'total_rec_late_fee': 'RECEIVED_LATE_FEE',
    'application_type': 'APPLICATION_TYPE',
    'tot_coll_amt': 'TOTAL_COLLECTION_AMOUNT',
    'tot_cur_bal': 'TOTAL_CURRENT_BALANCE',
    'loan_status': 'LOAN_STATUS'
})

df_new.to_csv('EDA_data.csv', index=False)

#sub data frame
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/mahesh/Library/CloudStorage/OneDrive-CharlesDarwinUniversity/Workspace/EDA/EDA_data.csv')

df.shape 
df.info()  
df.head()


#CHECK FOR NULL VALUES WITH COUNT
null_cols = df.columns[df.isnull().any()]
null_df = df[null_cols].isnull().sum().to_frame(name='Null Count')\
          .merge(df[null_cols].isnull().mean().mul(100).to_frame(name='Null Percent'), left_index=True, right_index=True)
null_df_sorted = null_df.sort_values(by='Null Count', ascending=False)
print(null_df_sorted)


distinct_entries = pd.Series(df['BANK_INVESTMENT'].value_counts()).sort_values(ascending=True)
# print sorted unique values
print(distinct_entries)

distinct_entries = pd.Series(df['BANK_INVESTMENT']).sort_values(ascending=True)
# print sorted unique values
print(distinct_entries)

#drop the column
df.drop(['EMPPLOYMENT_TITLE'], axis=1, inplace=True)


#CHANGING THE DATATYPE
df['TERM'] = df['TERM'].str.replace('months', '')
df['TERM'] = df['TERM'].astype(int)





df['EMPLOYEMENT_DURATION'] = df['EMPLOYEMENT_DURATION'].str.replace('years', '')
df['EMPLOYEMENT_DURATION'] = df['EMPLOYEMENT_DURATION'].str.replace('year', '')
df['EMPLOYEMENT_DURATION'] = df['EMPLOYEMENT_DURATION'].str.replace('+', '')    # In our analysis we will consider 10 as 10+ years 
df['EMPLOYEMENT_DURATION'] = df['EMPLOYEMENT_DURATION'].str.replace('< 1', '0') # In our analysis we will consider 0 as less than a year
df['EMPLOYEMENT_DURATION'] = df['EMPLOYEMENT_DURATION'].fillna('-1')
df['EMPLOYEMENT_DURATION'] = df['EMPLOYEMENT_DURATION'].astype(int)



#CHECK FOR NULL VALUES WITH COUNT
null_cols = df.columns[df.isnull().any()]
null_df = df[null_cols].isnull().sum().to_frame(name='Null Count')\
          .merge(df[null_cols].isnull().mean().mul(100).to_frame(name='Null Percent'), left_index=True, right_index=True)
null_df_sorted = null_df.sort_values(by='Null Count', ascending=False)
print(null_df_sorted)


df.to_csv('EDA_ready_data.csv', index=False)


#sub data frame
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/mahesh/Library/CloudStorage/OneDrive-CharlesDarwinUniversity/Workspace/EDA/EDA_ready_data.csv')

column_category=df.select_dtypes(include=['object']).columns
print(column_category)

column_numerical = df.select_dtypes(include=np.number).columns.tolist()
print(column_numerical)

df.describe(include='all').T
df.describe().T


#Histogram and box plots
fig, axs = plt.subplots(ncols=2, figsize=(10,5))
sns.histplot(df, x="BANK_INVESTMENT", bins=20,  color='purple',kde=False, ax=axs[0])
#axs[0].set_title('Histogram of BANK_INVESTMENT')
sns.boxplot(df, x="BANK_INVESTMENT", color='purple', ax=axs[1])
#axs[1].set_title('Boxplot of BANK_INVESTMENT')
plt.show()

fig, axs = plt.subplots(ncols=2, figsize=(10,5))
sns.histplot(df, x="INTEREST_RATE", bins=20, color='purple',kde=False, ax=axs[0])
#axs[0].set_title('Histogram of BANK_INVESTMENT')
sns.boxplot(df, x="INTEREST_RATE",color='purple', ax=axs[1])
#axs[1].set_title('Boxplot of BANK_INVESTMENT')
plt.show()



fig, axes = plt.subplots(2, 2, figsize = (18, 18))
fig.suptitle('Bar plot for all categorical variables in the dataset')
sns.countplot(ax = axes[0, 0], x = 'STATUS', data = df, color = 'purple', 
              order = df['STATUS'].value_counts().index);
sns.countplot(ax = axes[0, 1], x = 'LOAN_PURPOSE', data = df, color = 'purple', 
              order = df['LOAN_PURPOSE'].value_counts().index[:5]);
sns.countplot(ax = axes[1, 0], x = 'STATE', data = df, color = 'purple', 
              order = df['STATE'].value_counts().index[:20]);
sns.countplot(ax = axes[1, 1], x = 'INITIAL_LIST_STATUS', data = df, color = 'purple', 
              order = df['INITIAL_LIST_STATUS'].value_counts().index);
axes[1][1].tick_params(labelrotation=45);




# Select columns for pair plot
# Create pair plot
sns.pairplot(df[column_numerical])
plt.show()




fig, axarr = plt.subplots(3, 2, figsize=(12, 18))
df.groupby('STATUS')['BANK_INVESTMENT'].mean().sort_values(ascending=False).plot.bar(ax=axarr[0][0], fontsize=12,color='purple')
axarr[0][0].set_title("Status Vs Bank Investment(Average)", fontsize=18)
axarr[0][0].set_xticklabels(axarr[0][0].get_xticklabels(), rotation=0)

df.groupby('LOAN_PURPOSE')['BANK_INVESTMENT'].mean().sort_values(ascending=False).plot.bar(ax=axarr[0][1], fontsize=12,color='purple')
axarr[0][1].set_title("Loan Purpose Vs Bank Investment(Average)", fontsize=18)

df.groupby('STATE')['BANK_INVESTMENT'].mean().sort_values(ascending=False).head(10).plot.bar(ax=axarr[1][0], fontsize=12,color='purple')
axarr[1][0].set_title("State vs top 5 States by Bank Investment(Average)", fontsize=18)
axarr[1][0].set_xticklabels(axarr[1][0].get_xticklabels(), rotation=0)

df.groupby('INITIAL_LIST_STATUS')['BANK_INVESTMENT'].mean().sort_values(ascending=False).plot.bar(ax=axarr[1][1], fontsize=12,color='purple')
axarr[1][1].set_title("Initial list status Vs Bank Investment(Average)", fontsize=18)
axarr[1][1].set_xticklabels(axarr[1][1].get_xticklabels(), rotation=0)

df.groupby('APPLICATION_TYPE')['BANK_INVESTMENT'].mean().sort_values(ascending=True).head(10).plot.bar(ax=axarr[2][0], fontsize=12,color='purple')
axarr[2][0].set_title("Application type Vs Bank Investment(Average)", fontsize=18)
axarr[2][0].set_xticklabels(axarr[2][0].get_xticklabels(), rotation=0)

df.groupby('TERM')['BANK_INVESTMENT'].mean().sort_values(ascending=False).head(10).plot.bar(ax=axarr[2][1], fontsize=12,color='purple')
axarr[2][1].set_title("Term Vs Bank Investment(Average) ", fontsize=18)
axarr[2][1].set_xticklabels(axarr[2][1].get_xticklabels(), rotation=0)

plt.subplots_adjust(hspace=1.0)
plt.subplots_adjust(wspace=.5)
sns.despine()


#Heat map
plt.figure(figsize=(12, 7))
sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, cmap="Purples")
plt.show()


#Imputation
df[['TOTAL_COLLECTION_AMOUNT','TOTAL_CURRENT_BALANCE','ANNUAL_INCOME']].mean()

#replaced with mean value
df['TOTAL_COLLECTION_AMOUNT'].fillna(df['TOTAL_COLLECTION_AMOUNT'].mean(), inplace=True)

#replaced with minimum value
df['TOTAL_CURRENT_BALANCE'].fillna(df['TOTAL_CURRENT_BALANCE'].min(), inplace=True)
df['ANNUAL_INCOME'].fillna(df['ANNUAL_INCOME'].min(), inplace=True)


# REMOVING OUTLIERS

#10 highest count values along with their respective unique values in a single line:
print(df['INTEREST_RATE'].value_counts().sort_index(ascending=False)[:10])

count_high_interest_rate = len(df[df['INTEREST_RATE'] > 10])
print("Number of rows with interest rate > 10:", count_high_interest_rate)



def find_outliers(df, col):
    Q1 = df[col].describe()['25%']
    Q3 = df[col].describe()['75%']
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return outliers

outliers = find_outliers(df, 'ANNUAL_INCOME')
print(outliers)

outliers = find_outliers(df, 'INTEREST_RATE')
df.drop(outliers.index, inplace=True)



distinct_entries = pd.Series(df['BANK_INVESTMENT'].value_counts()).sort_values(ascending=True)
# print sorted unique values
print(distinct_entries)
# get sorted unique values of 'currency' and other_currency column
distinct_entries = pd.Series(df['STATE'].unique()).sort_values()
# print sorted unique values
print(distinct_entries)




print(df.loc[df['BANK_INVESTMENT'] < 5, ['BANK_INVESTMENT', 'STATE']])

###Visualization
df['TERM'] = df['TERM'].str.replace('36', '36 months')
df['TERM'] = df['TERM'].str.replace('60', '60 months')
df['TERM'] = df['TERM'].astype('string')

df['LOAN_STATUS'] = df['LOAN_STATUS'].astype('string')
df['LOAN_STATUS'] = df['LOAN_STATUS'].str.replace('0', 'paid')
df['LOAN_STATUS'] = df['LOAN_STATUS'].str.replace('1', 'default')

#Giving unique id
df = df.reset_index().rename(columns={'index': 'ID'})
df.to_csv('final_eda.csv', index='False')
df.to_excel('final_eda.xlsx', index='False')