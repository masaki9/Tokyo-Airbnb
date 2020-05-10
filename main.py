# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from math import sqrt
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# pd.set_option('display.max_rows', None)

data_reviews = "data/reviews.csv"
data_listings = "data/listings.csv"
data_calendar = "data/calendar.csv"

df_reviews = pd.read_csv(data_reviews, index_col=0, sep=',')
df_listings = pd.read_csv(data_listings, index_col=0, sep=',')
df_calendar = pd.read_csv(data_calendar, index_col=0, sep=',')

# %%
df_listings.head(5)

# %% How many datapoints in the listings dataset?
print("How many datapoints in the listings dataset?")
df_listings.shape
# (15551, 105)

# %%
print("How many datapoints in the reviews dataset?")
df_reviews.shape
# (416394, 5)

# %% How many missing values does the listing dataset have?
print("How many missing values does the listing dataset have?")
df_listings.isnull().sum().sum()
# 187136 missing values

# %% Clean Data - Remove Unnecessary Symbols
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


# %% Which columns have the most missing values?
print("Which columns have the most missing values?")

def missing_data(df):
    total = df.isnull().sum()
    percent = (df.isnull().sum()/df.isnull().count()*100)
    missing_values = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        types.append(dtype)

    missing_values['Types'] = types
    missing_values.sort_values('Total', ascending=False, inplace=True)

    return np.transpose(missing_values)

missing_data(df_listings)

x = missing_data(df_listings)
print(x)

# The following columns have 98+ percent rate of missing values.
# thumbnail_url neighbourhood_group_cleansed jurisdiction_names xl_picture_url medium_url square_feet monthly_price weekly_price

# %% Let's plot these missing values(%) vs column_names. Top 10 Features with the most missing values.
missing_values_count = (df_listings.isnull().sum()/df_listings.isnull().count()*100).sort_values(ascending=False)
plt.figure(figsize=(20,10))
base_color = sns.color_palette()[0]
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
sns.barplot(missing_values_count[:10].index.values, missing_values_count[:10], color=base_color)
plt.show()

# %% Let's drop features with high missing data rate.
# The following columns have 98+ percent rate of missing values.
# thumbnail_url neighbourhood_group_cleansed jurisdiction_names xl_picture_url medium_url square_feet monthly_price weekly_price

columns_to_drop = ['thumbnail_url', 'neighbourhood_group_cleansed', 'jurisdiction_names', 'xl_picture_url', 'medium_url', 'square_feet', 'monthly_price', 'weekly_price']
df_listings.drop(columns_to_drop, axis=1, inplace=True)

# %% Summary Statistics
print()

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

print("\nAverage Prices by Neighborhood")
for i in range(len(neighbourhoods)):
    print(neighbourhoods[i] + ": {:0.0f} Yen".format(avg_neigh_prices[i]))

df_avg_neigh_prices = pd.DataFrame(list(zip(neighbourhoods, avg_neigh_prices)), columns=['neighbourhood', 'price']).sort_values(by='price', ascending=False)

plt.figure(figsize=(16,8))
plt.bar(df_avg_neigh_prices['neighbourhood'], df_avg_neigh_prices['price'], color='lightblue')
plt.xticks(rotation='vertical', fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Neighbourhood", fontsize=14)
plt.ylabel('Price (Yen)', fontsize=14)
plt.title('Average Price of Tokyo Airbnb by Neighbourhood', fontsize=20)
plt.show()

# %% Top 10 Neighbourhood by Average Price
df_avg_neigh_prices = pd.DataFrame(list(zip(neighbourhoods, avg_neigh_prices)), columns=['neighbourhood', 'price']).sort_values(by='price', ascending=False)

plt.figure(figsize=(16,10))
plt.bar(df_avg_neigh_prices['neighbourhood'][:10], df_avg_neigh_prices['price'][:10], color='lightblue')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Neighbourhood", fontsize=14)
plt.ylabel('Price (Yen)', fontsize=14)
plt.title('Top 10 Neighbourhoods by Average Price', fontsize=20)
plt.show()

# %% Popular Neighbourhoods in Tokyo
# Top 10 Neighbourhoods by # of bookings
counts = df_listings["neighbourhood_cleansed"].value_counts()
neigh_counts = counts.tolist()
neighbourhoods = df_listings["neighbourhood_cleansed"].value_counts().index.tolist()
df_popular_neighbourhoods = pd.DataFrame(list(zip(neighbourhoods, neigh_counts)),
                                         columns=['neighbourhood', 'num_bookings'])

plt.figure(figsize=(16,10))
plt.bar(df_popular_neighbourhoods['neighbourhood'][:10], df_popular_neighbourhoods['num_bookings'][:10], color='lightblue')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Neighbourhood", fontsize=14)
plt.ylabel('Number of Bookings', fontsize=14)
plt.title('Top 10 Neighbourhoods by Number of Bookings', fontsize=20)
plt.show()

# %% Box Plot
base_color = sns.color_palette()[0]
plt.figure(figsize=(40,15))
plt.xticks(rotation=45)
plt.yticks(np.arange(0, 1200000, step=50000))
sns.boxplot(data=df_listings, x='neighbourhood_cleansed', y='price', color=base_color)\
    .set(ylabel='Price (Yen)', xlabel='Neighbourhood')
plt.show()

# %% Price Distribution Plot
hist_kws={"alpha": 0.3}
plt.figure(figsize=(30,10))
plt.xticks(np.arange(0, 1000000, step=25000))
sns.distplot(df_listings['price'], hist_kws=hist_kws).set(xlabel='Price (Yen)')
plt.show()

# %% Frequency Histogram
plt.figure(figsize=(30,10))
plt.xticks(np.arange(0, 1000000, step=25000))
plt.hist(df_listings['price'], bins=500)
plt.xlabel("Price (Yen)")
plt.show()

# %% Distribution of Types of Rooms
print()
print(df_listings["room_type"].value_counts())

num_entire = np.where(df_listings["room_type"] == "Entire home/apt")[0].size
num_private = np.where(df_listings["room_type"] == "Private room")[0].size
num_shared = np.where(df_listings["room_type"] == "Shared room")[0].size
num_hotel_room = np.where(df_listings["room_type"] == "Hotel room")[0].size

labels = 'Entire Home/Apt', 'Private Room', 'Shared Room', 'Hotel Room'
sizes = [num_entire, num_private, num_shared, num_hotel_room]
colors = ['lightpink','lightblue','lightgreen','beige']

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%',  colors=colors, shadow=True, startangle=90, pctdistance=0.8)
plt.rcParams['font.size'] = 14
plt.title("Distribution of Types of Rooms", fontsize=20)

# Draw white circle
centre_circle = plt.Circle((0,0), 0.60, fc='white')
fig1 = plt.gcf()
fig1.gca().add_artist(centre_circle)

plt.show()

# %% Distribution of Cancellation Policies
print()
print(df_listings["cancellation_policy"].value_counts())

num_strict = np.where((df_listings["cancellation_policy"] == "strict")
                      | (df_listings["cancellation_policy"] == "strict_14_with_grace_period")
                      | (df_listings["cancellation_policy"] == "super_strict_30")
                      | (df_listings["cancellation_policy"] == "super_strict_60"))[0].size

num_moderate = np.where(df_listings["cancellation_policy"] == "moderate")[0].size
num_flexible = np.where(df_listings["cancellation_policy"] == "flexible")[0].size

labels = 'Strict', 'Moderate', 'Flexible'
sizes = [num_strict, num_moderate, num_flexible]
colors = ['lightpink','lightblue','lightgreen','beige']

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, shadow=True, startangle=90, pctdistance=0.8)
plt.rcParams['font.size'] = 14
plt.title("Distribution of Cancellation Policies", fontsize=20)

# Draw white circle
centre_circle = plt.Circle((0,0), 0.60, fc='white')
fig1 = plt.gcf()
fig1.gca().add_artist(centre_circle)

plt.show()


# %% Describe numerical features
df_listings.describe()

# %% Plotting Histograms of Each Variable
df_listings.hist(bins=50, figsize=(40,30))
plt.tight_layout(pad=0.4)
plt.show()

# %% Feature Engineering
print("\nFeature Engineering")

print("Shape Before Removing Outliers: {}".format(df_listings.shape))

# Remove outliers and print stats
# df_listings_no_outliers = df_listings[df_listings['price'] < 110000]
# df_listings_no_outliers = df_listings[df_listings['price'] < 100000]
# df_listings_no_outliers = df_listings[df_listings['price'] < 50000]
df_listings_no_outliers = df_listings[df_listings['price'] < 48000]

# Remove unnecessary ID columns
df_listings_no_outliers.drop('host_id', axis=1, inplace=True)
df_listings_no_outliers.drop('scrape_id', axis=1, inplace=True)

# Fill in the missing values for numerical columns.
# Get numerical columns
df_listings_no_outliers = df_listings_no_outliers.select_dtypes(exclude=['object'])

# I am using median values to fill in the missing values.
df_listings_no_outliers.fillna(df_listings_no_outliers.median(), inplace=True)

print("Shape After Removing Outliers: {}".format(df_listings_no_outliers.shape))
# print(df_listings_no_outliers.shape)
# (14851, 37) < 50000
# (14188, 37) # < 48000

# %% Summary Statistics After Removing Outliers
print()

avg_price2 = np.mean(df_listings_no_outliers["price"])
print("Average Listing Price: {} Yen".format(avg_price2))

max_price2 = np.max(df_listings_no_outliers["price"])
print("Maximum Listing Price: {} Yen".format(max_price2))

min_price2 = np.min(df_listings_no_outliers["price"])
print("Minimum Listing Price: {} Yen".format(min_price2))

# %% What are the most correlated features for prices?
# corr_matrix = df_listings.corr()
corr_matrix = df_listings_no_outliers.corr()

plt.subplots(figsize=(30,20))
# sns.heatmap(corr_matrix, xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns,
#             vmax=1.0, square=True, cmap="Blues")

# with annotation
sns.heatmap(corr_matrix, xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns,
            vmax=1.0, square=True, cmap="Blues", annot=True, fmt='.2f')

plt.show()

print("\nWhat are the most correlated features for prices?")
# Get the top 10 most correlated features for prices
# corr_matrix = df_listings.corr()
print(corr_matrix['price'].sort_values(ascending=False)[0:11])

# %% corr() only captures linear relationships, so it's not the most reliable way to detect correlations.
# So let's plot some of the price-related features on a scatter plot matrix.

# Before Removing Outliers
# attributes = ["price", "host_listings_count", "host_total_listings_count", 'calculated_host_listings_count_entire_homes',
#               'calculated_host_listings_count', "accommodates", "guests_included", "extra_people",
#               "bedrooms", "cleaning_fee", "availability_365"]
# scatter_matrix(df_listings[attributes], figsize=(40, 40))

# After Removing Outliers
attributes = ["price", "accommodates", "guests_included", 'cleaning_fee',
              "beds", "bedrooms"]

scatter_matrix(df_listings_no_outliers[attributes], figsize=(20, 20))
plt.show()

# %%
# features = ["accommodates", "guests_included", 'cleaning_fee', "beds", "bedrooms"]
# features = ["accommodates", "guests_included", 'cleaning_fee', "beds", "bedrooms", "extra_people"]
# features = ["accommodates", "guests_included", 'cleaning_fee', "beds", "bedrooms", "extra_people",
#             "review_scores_rating", "reviews_per_month", "review_scores_cleanliness", "security_deposit"]
# X = df_listings_no_outliers[features]

# I have found that including all 39 numerical features gives the best adjusted R score.
X = df_listings_no_outliers.drop('price', axis=1)
y = df_listings_no_outliers[['price']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# %% Adjusted R Squared Function
def adjusted_r_squared(X, y, model):
    R2 = model.score(X, y)
    n = len(y)
    k = X.shape[1]
    return 1 - (1 - R2) * (n - 1) / (n - k - 1)

# %% Linear Regression
print("\nLinear Regression")
linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

# y_train_pred = linear_model.predict(X_train)
y_test_pred = linear_model.predict(X_test)

# Calculate deviation between actual and predicted values.
rmse = sqrt(mean_squared_error(y_test, y_test_pred))
print("The root mean square error calculation is: {}".format(rmse))

# print("Intercept: {}".format(linear_model.intercept_))
# print("Coefficients: {}".format(linear_model.coef_))

# Return the coefficient of determination R^2 of the prediction.
print("LM Score (R-Squared): {}".format(linear_model.score(X, y)))
# LM Score (R-Squared): approx. 0.4430 (with every numerical feature)
# LM Score (R-Squared): approx. 0.3975 (with top 6 price-related features)
# LM Score (R-Squared): approx. 0.4074 (with top 10 price-related features)

print("Adjusted R-Squared: {}".format(adjusted_r_squared(X, y, linear_model)))
# Adjusted R-Squared: approx. 0.4423 (with every numerical feature)
# Adjusted R-Squared: approx. 0.3974 (with top 6 price-related features)
# Adjusted R-Squared: approx. 0.4070 (with top 10 price-related features)

# Find the prediction R2 of RandomForestRegressor model using 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True)
rf_cross_val = cross_val_score(linear_model, X, y, cv=kfold, scoring='r2')
print(rf_cross_val)
mean_cross_val_score = rf_cross_val.mean()
print("The mean R2 score using 5-fold cross-validation is: {}".format(mean_cross_val_score))

# Create a dataframe that has actual prices and predicted prices
df_lm_test = pd.DataFrame({'Actual': y_test['price']})
df_lm_test['Prediction'] = y_test_pred

# Compare the first actual and predicted prices.
df_lm_test.head(20)

# %% Random Forest Regressor
print("\nRandom Forest Regressor")

rf_model = RandomForestRegressor(n_estimators=100, criterion='mse', bootstrap=True)

rf_model.fit(X_train, y_train)

y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Calculate deviation between actual and predicted values.
rmse = sqrt(mean_squared_error(y_test, y_test_pred))
print("The root mean square error calculation is: {}".format(rmse))

print("RF Model Score: {}".format(rf_model.score(X, y)))
# RF Model Score: 0.8704147620744445

print("RF Model Score (Adjusted): {}".format(adjusted_r_squared(X, y, rf_model)))
# RF Model Score (Adjusted): 0.8700759172826956

# Find the prediction R2 of RandomForestRegressor model using 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True)
rf_cross_val = cross_val_score(rf_model, X, y, cv=kf, scoring='r2')
print(rf_cross_val)
mean_cross_val_score = rf_cross_val.mean()
print("The mean R2 score using 5-fold cross-validation is: {}".format(mean_cross_val_score))
# The mean R2 score using 5-fold cross-validation is: 0.7130852814443622

# Create a dataframe that has actual prices and predicted prices
df_rf_test = pd.DataFrame({'Actual': y_test['price']})
df_rf_test['Prediction'] = y_test_pred

# Compare the first 20 actual and predicted prices.
df_rf_test.head(20)

# %% Feature Importances from the Random Forest Regression Model
values = sorted(zip(X_train.columns, rf_model.feature_importances_), key=lambda x: x[1] * -1)
feature_importances = pd.DataFrame(values, columns=["Name", "Score"])
feature_importances = feature_importances.sort_values(by=['Score'], ascending=False)

features = feature_importances['Name'][:10]
y_pos = np.arange(len(features))
scores = feature_importances['Score'][:10]

plt.figure(figsize=(10, 10))
plt.bar(y_pos, scores, align='center')
plt.xticks(y_pos, features, rotation='vertical')
plt.ylabel('Score')
plt.xlabel('Features')
plt.title('Top 10 Important Features (Random Forest Regression)')

plt.show()
