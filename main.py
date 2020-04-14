# %%
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# pd.set_option('display.max_rows', None)

data_reviews = "data/reviews.csv"
data_listings = "data/listings.csv"
data_calendar = "data/calendar.csv"

df_reviews = pd.read_csv(data_reviews, index_col=0, sep=',')
df_listings = pd.read_csv(data_listings, index_col=0, sep=',')

# %%
df_listings.head(5)

# %% How many datapoints in the listings dataset?
df_listings.shape
# (15551, 105)

# %% How many missing values does the listing dataset have?
df_listings.isnull().sum().sum()
# 187136 missing values

# %% Clean Data
def remove_unnecessary_symbols(df, column):
    # Remove the dollar signs and commas in the column.
    df[column] = df_listings[column].astype(str)
    df[column] = df[column].str.replace('$', '')
    df[column] = df[column].str.replace(',', '')
    df[column] = df[column].astype(float)
    return df[column]


df_listings['price'] = remove_unnecessary_symbols(df_listings, 'price')
df_listings['weekly_price'] = remove_unnecessary_symbols(df_listings, 'weekly_price')
df_listings['monthly_price'] = remove_unnecessary_symbols(df_listings, 'monthly_price')
df_listings['extra_people'] = remove_unnecessary_symbols(df_listings, 'extra_people')
df_listings['security_deposit'] = remove_unnecessary_symbols(df_listings, 'security_deposit')
df_listings['cleaning_fee'] = remove_unnecessary_symbols(df_listings, 'cleaning_fee')


# %% Which columns have the most missing values?
def missing_data(df):
    total = df.isnull().sum()
    percent = (df.isnull().sum()/df.isnull().count()*100)
    missing_values = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        types.append(dtype)
    missing_values['Types'] = types
    missing_values.sort_values('Total',ascending=False,inplace=True)
    return(np.transpose(missing_values))

missing_data(df_listings)
# The following columns have 98+ percent rate of missing values.
# thumbnail_url neighbourhood_group_cleansed jurisdiction_names xl_picture_url medium_url square_feet monthly_price weekly_price

# %% Let's plot these missing values(%) vs column_names. Top 10 Features with the most missing values.
missing_values_count = (df_listings.isnull().sum()/df_listings.isnull().count()*100).sort_values(ascending=False)
plt.figure(figsize=(20,10))
base_color = sns.color_palette()[0]
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
sns.barplot(missing_values_count[:10].index.values, missing_values_count[:10], color = base_color)
plt.show()

# %% Describe numerical features
df_listings.describe()

# %%
df_listings.hist(bins=50, figsize=(40,30))
plt.tight_layout(pad=0.4)
plt.show()

# %% Describe categorical features
df_listings.describe(include='O')

# %% What are the most correlated features?
corr_matrix = df_listings.corr()
plt.subplots(figsize=(30,20))
# sns.heatmap(corr_matrix, xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns,
#             vmax=1.0, square=True, cmap="Blues")

# with annotation
sns.heatmap(corr_matrix, xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns,
            vmax=1.0, square=True, cmap="Blues", annot=True, fmt='.2f')

plt.show()
