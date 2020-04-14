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


columns_to_clean = ['price', 'weekly_price', 'monthly_price', 'extra_people', 'security_deposit', 'cleaning_fee']

for column in columns_to_clean:
    df_listings[column] = remove_unnecessary_symbols(df_listings, column)

# df_listings['price'] = remove_unnecessary_symbols(df_listings, 'price')
# df_listings['weekly_price'] = remove_unnecessary_symbols(df_listings, 'weekly_price')
# df_listings['monthly_price'] = remove_unnecessary_symbols(df_listings, 'monthly_price')
# df_listings['extra_people'] = remove_unnecessary_symbols(df_listings, 'extra_people')
# df_listings['security_deposit'] = remove_unnecessary_symbols(df_listings, 'security_deposit')
# df_listings['cleaning_fee'] = remove_unnecessary_symbols(df_listings, 'cleaning_fee')

# %% Summary Statistics
avg_price = np.mean(df_listings["price"])
print("Average Listing Price: {} Yen".format(avg_price))

max_price = np.max(df_listings["price"])
print("Maximum Listing Price: {} Yen".format(max_price))

min_price = np.min(df_listings["price"])
print("Minimum Listing Price: {} Yen".format(min_price))

# Average Listing Price: 23982.066169378177 Yen
# Maximum Listing Price: 1063924.0 Yen
# Minimum Listing Price: 0.0 Yen

# %% Average Price by Neighbourhood
# Determine most expensive neighborhood on average
neighbourhoods = df_listings['neighbourhood_cleansed'].unique()
avg_neigh_prices = np.zeros(len(neighbourhoods))

for neighbourhood in range(len(neighbourhoods)):
    list_of_neigh_price = df_listings.loc[df_listings['neighbourhood_cleansed'] == neighbourhoods[neighbourhood], 'price']
    avg_price_single_neigh = np.mean(list_of_neigh_price)
    avg_neigh_prices[neighbourhood] = avg_price_single_neigh

print("Average Prices by Neighborhood\n")

for i in range(len(neighbourhoods)):
    print(neighbourhoods[i] + ": {:0.0f} Yen".format(avg_neigh_prices[i]))

fig, ax = plt.subplots()
rects = ax.bar(neighbourhoods, avg_neigh_prices)

ax.set_xlabel("Neighbourhood", fontsize=14)
ax.set_ylabel('Price', fontsize=14)
ax.set_title('Average Price of Tokyo Airbnb by Neighbourhood', fontsize=20)
plt.xticks(rotation='vertical', fontsize=14)
plt.yticks(fontsize=14)
plt.rcParams['figure.figsize'] = (16,8)
plt.show()

# %% Popular Neighbourhoods in Tokyo

# Find the most popular neighborhoods.
neighbourhood_counts = df_listings["neighbourhood_cleansed"].value_counts()

# print(df_listings["neighbourhood_cleansed"].unique().tolist())
print("Number of Tokyo Neighbourhoods: {}".format(len(df_listings["neighbourhood_cleansed"].unique().tolist())))

print(neighbourhood_counts)

# Small workaround here since the indexing was weird in the provided data structure
neighbourhoods = np.asarray(neighbourhood_counts.axes)[0]
counts = neighbourhood_counts.values

fig, ax = plt.subplots()
rects = ax.bar(neighbourhoods, counts)
ax.set_xlabel("Neighborhood")
ax.set_ylabel("Number of rentals")
ax.set_title("Number of Bookings per Neighborhood")
plt.xticks(rotation='vertical')
plt.show()

# Top 5 neighbourhoods are Shinjuku, Taito, Toshima, Sumida, and Shibuya.

# %% Distribution of Types of Rooms
print(df_listings["room_type"].value_counts())

num_entire = np.where(df_listings["room_type"] == "Entire home/apt")[0].size
num_private = np.where(df_listings["room_type"] == "Private room")[0].size
num_shared = np.where(df_listings["room_type"] == "Shared room")[0].size
num_hotel_room = np.where(df_listings["room_type"] == "Hotel room")[0].size

labels = 'Entire Home/Apt', 'Private Room', 'Shared Room', 'Hotel Room'
sizes = [num_entire, num_private, num_shared, num_hotel_room]

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Distribution of Types of Rooms")
plt.show()

# %% Distribution of Cancellation Policies
print(df_listings["cancellation_policy"].value_counts())

num_strict = np.where((df_listings["cancellation_policy"] == "strict")
                      | (df_listings["cancellation_policy"] == "strict_14_with_grace_period")
                      | (df_listings["cancellation_policy"] == "super_strict_30")
                      | (df_listings["cancellation_policy"] == "super_strict_60"))[0].size

num_moderate = np.where(df_listings["cancellation_policy"] == "moderate")[0].size
num_flexible = np.where(df_listings["cancellation_policy"] == "flexible")[0].size

labels = 'Strict', 'Moderate', 'Flexible'
sizes = [num_strict, num_moderate, num_flexible]

# color scheme from https://medium.com/@kvnamipara/a-better-visualisation-of-pie-charts-by-matplotlib-935b7667d77f
# colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

fig, ax = plt.subplots()
# ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, shadow=True, startangle=90)
ax.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
# ax.axis('equal')
plt.title("Distribution of Cancellation Policies")
plt.show()

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
