
# Import argument parser
import argparse

#Import The Libraries
import warnings
import sys
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Scalers
from sklearn.preprocessing import StandardScaler


# Models for classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

from imblearn.over_sampling import SMOTE

# Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Parser arguments
parser = argparse.ArgumentParser(description='Settings')

parser.add_argument('--db_path', default = './data/fishing.db', type=str,
                    help = "Database directory: default = './data/fishing.db'")

# ------------------------------------------------------------------------------
# Read Data in Pandas Dataframe
# ------------------------------------------------------------------------------
# Setup connection and import data
db_path = parser.parse_args().db_path
conn = sqlite3.connect(db_path)
data = pd.read_sql_query(sql="SELECT * FROM fishing", con=conn)
data.head()

data.shape

# -----------------------------------------------------------------
# DATA CLEANING
# -----------------------------------------------------------------

# Check for duplicate rows
duplicate_rows = data[data.duplicated()]

# Print the result
print("Duplicate Rows (excluding first occurrence):")
print(duplicate_rows)

# Remove duplicate rows
data = data.drop_duplicates()

# Display the data without duplicates
print(data)

data.shape

data.dtypes

# -----------------------------------------------------------------
# Identifying numerical, discrete and continuous columns
# -----------------------------------------------------------------
num_feats = data.select_dtypes(include=['float64']).columns.tolist()

# identify discrete features
discrete_feats = []
for feat in num_feats:
    if data[feat].nunique() < 10:
        discrete_feats.append(feat)

# identify continuous features
continuous_feats = list(set(num_feats) - set(discrete_feats))

print("Numerical Features:", num_feats)
print("Discrete Features:", discrete_feats)
print("Continuous Features:", continuous_feats)

# -----------------------------------------------------------------
# Invalid Data
# -----------------------------------------------------------------
# Check for invalid data for categorical and discrete column
# To display the labels in each categorical and discrete columns
column_names = [
    'WindGustDir', 'WindDir9am', 'WindDir3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm'
]

for column in column_names:
    categories = set(filter(lambda x: not np.isnan(x) if isinstance(x, float) else True, data[column]))
    print(f"Categories for {column}:")
    for category in categories:
        print(f" - {category}")
    print()

# -----------------------------------------------------------------
# Change Pressure Data columns To Uppercase for consistency
# -----------------------------------------------------------------
data['Pressure9am'] = data['Pressure9am'].str.upper()
data['Pressure3pm'] = data['Pressure3pm'].str.upper()

# To check for non-numerical data in each numerical-continuous columns
column_names = [
   'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'AverageTemp'
]

def is_non_numerical(value):
    return not isinstance(value, (int, float))

for column in column_names:
    non_numerical_data = list(filter(is_non_numerical, data[column]))
    print(f"Non-numerical data for {column}:")
    for value in non_numerical_data:
        print(f" - {value}")
    print()

# -----------------------------------------------------------------
# There are negative hour for sunshine, this is erroneous data
# Convert the sunshine values to absolute
# -----------------------------------------------------------------
# Convert the negative value in the Sunshine column to positive
data['Sunshine'] = data['Sunshine'].abs()

# -----------------------------------------------------------------
# Missing Data
# -----------------------------------------------------------------
# Count missing data for each column
missing_data_count = data.isnull().sum()

# Print the count of missing data for each column
print(missing_data_count)

# -----------------------------------------------------------------------------------------------------------
# From above data the number of missing data for each column is not significant (less than 3%)
# Fill the miss data with mean or median depending on the distribution curve, or mode if it is discrete data
# Intend to remove RainToday column as it is redundant since it is a function of Rainfall column. 
# Therefore no need to handle the missing data for RainToday.
# -----------------------------------------------------------------------------------------------------------


# Create bar plot for each discrete feature
for feat in discrete_feats:
    plt.figure()
    data[feat].value_counts().plot(kind='bar')
    plt.title(feat)
    plt.xlabel('Value')
    plt.ylabel('Count')

plt.show()

# Create histogram for each continuous feature
for feat in continuous_feats:
    plt.figure()
    plt.hist(data[feat], bins=20)
    plt.title(feat)
    plt.xlabel('Value')
    plt.ylabel('Count')
plt.show()

# Get summary statistics for each numerical column
summary_statistics = data.describe()

# Print summary statistics
print(summary_statistics)

# ---------------------------------------------------------------------------------------------------------------------------------
# Fill column (Humidity3pm) missing data with mean because the distibution shown for Humidity3pm is normally distributed
# Although the distribution curve shown for Humidity9am is slightly skewed but to maintain consistency with the Humidity3pm column,
# mean is used instead of median to fill the missing data.
# ---------------------------------------------------------------------------------------------------------------------------------
# Calculate the mean of Humidity3pm and fill missing data with the mean value.
col_mean = int(data['Humidity3pm'].mean())
data['Humidity3pm'].fillna(col_mean, inplace=True)

# Calculate the mean of Humidity9am and fill missing data with the mean value.
col_mean = int(data['Humidity9am'].mean())
data['Humidity9am'].fillna(col_mean, inplace=True)

# ----------------------------------------------------------------------------------------------------
# Fill numerical columns missing data with median. Median is used because the distibution shown above is skewed
# ----------------------------------------------------------------------------------------------------
# List of specified column names containing continuous numerical data with missing values.
column_names = [
    'WindGustSpeed', 'WindSpeed3pm', 'Sunshine', 'Evaporation', 'Humidity3pm', 'Rainfall', 'Humidity9am', 'WindSpeed9am', 'AverageTemp'
]

# Calculate the mean and fill missing values for each column
for column in column_names:
    if data[column].dtype in ['int64', 'float64']:  # Check if the column contains numerical data
          col_mean = round(data[column].mean(), 1)  # Calculate the mean and round to one decimal place
          data[column].fillna(col_mean, inplace=True)

# --------------------------------------------------------------------------------------------------------
# Fill categorical and discrete column missing data with mode.
# --------------------------------------------------------------------------------------------------------
# List of specified column names containing discrete numerical and categorical data with missing values.
column_names = [
    'WindGustDir', 'WindDir3pm', 'WindDir9am', 'Pressure3pm', 'Pressure9am', 'Cloud3pm', 'Cloud9am'
]

# Calculate the mean and fill missing values for each column
for column in column_names:
    col_mode = data[column].mode()
    data[column].fillna(col_mode, inplace=True)


# -----------------------------------------------------------------
# Outliers
# -----------------------------------------------------------------
# Determine number of outliers
# List of numeric column names
column_names = [
    'WindGustSpeed', 'WindSpeed3pm', 'Sunshine', 'Evaporation', 'Humidity3pm', 'Rainfall', 'Humidity9am', 'WindSpeed9am', 'AverageTemp', 'Cloud3pm', 'Cloud9am'
]

# Function to count outliers using the IQR method
def count_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return ((series < lower_bound) | (series > upper_bound)).sum()

# Count the number of outliers for each column
outliers_counts = {}
for column in column_names:
    if data[column].dtype in ['int64', 'float64']:  # Check if the column contains numerical data
        # outliers_count = count_outliers(data[column].dropna())  # Count outliers without missing values
        outliers_count = count_outliers(data[column])
        outliers_counts[column] = outliers_count

# Print the number of outliers for each column
for column, count in outliers_counts.items():
    print(f"{column}: {count} outliers")

# Note : Based on the outlier count above for each of the numerical data column,
# the total number of outliers is more than 30% of the dataset.
# This is significant, thus will not remove the outliers.

# ------------------------------------------------------------------------------
# CORRELATION ANALYSIS
# ------------------------------------------------------------------------------
# Select Numeric Columns
numeric_data = data.select_dtypes(include='number')

# Calculate the correlation matrix
corr_matrix = numeric_data.corr()

# Print the correlation matrix
print("Correlation matrix:")
print(corr_matrix)

# Print the column names
print("Column names:")
print(list(numeric_data.columns))

# Plot the correlation matrix
plt.figure(figsize = (8,8))
plt.matshow(corr_matrix, cmap = 'coolwarm', fignum = 1)
plt.colorbar()
plt.show()


# ------------------------------------------------------------------------------------------------------------
# Drop Unnecessary Column In Dataset
# Remove the Date and ColourOfBoats columns, as both columns are not relevancew.
# Remove the RainToday column, as it is redundant too
# as it defined by RainFall (rain is present when greater than or equal to 1.0, and absent when less than 1.0)
# -------------------------------------------------------------------------------------------------------------
X = data.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20]].values
Y = data.iloc[:,18].values

print(X)

Y = Y.reshape(-1,1) #1d list into 2d list 
print(Y)

imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
X = imputer.fit_transform(X)
Y = imputer.fit_transform(Y)

# ------------------------------------------------------------------------------
# ONE HOT ENCODING
# ------------------------------------------------------------------------------
le1 = LabelEncoder()
X[:,0] = le1.fit_transform(X[:,0])
le2 = LabelEncoder()
X[:,4] = le2.fit_transform(X[:,4])
le3 = LabelEncoder()
X[:,6] = le3.fit_transform(X[:,6])
le4 = LabelEncoder()
X[:,7] = le4.fit_transform(X[:,7])
le5 = LabelEncoder()
X[:,12] = le5.fit_transform(X[:,12])
le6 = LabelEncoder()
X[:,13] = le6.fit_transform(X[:,13])
le7 = LabelEncoder()
Y[:,0] = le7.fit_transform(Y[:,0])

print(X)

print(Y)

Y = np.array(Y,dtype=float)
print(Y)

# ------------------------------------------------------------------------------
# FEATURE SCALING
# ------------------------------------------------------------------------------
sc = StandardScaler()
X = sc.fit_transform(X)


# ------------------------------------------------------------------------------
# MODELLING
# ------------------------------------------------------------------------------

# check for imbalance data
# Count 'yes' and 'no' occurrences for dependent parameter
counts = data['RainTomorrow'].value_counts()
print("RainTomorrow :\n", counts)

# Handle imbalanced Data
oversample = SMOTE()
X, Y = oversample.fit_resample(X, Y)

# Splitting Dataset into Training set and Test set
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

print(X_train)

print(Y_train)

# ------------------------------------------------------------------------------
# Logistic Regression
# ------------------------------------------------------------------------------
# Create a Logistic Regression Model
lr = LogisticRegression()

# Train the model on the training data
lr.fit(X_train, np.ravel(Y_train))

# Make predictions on the testing data
Y_pred = lr.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(Y_test, Y_pred)

# Print the classification report
print('Classification report for Logistics Regression:')
print('')
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test, Y_pred ))
print(f'Accuracy: {accuracy:.2f}')

# ------------------------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------------------------
# Create a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
clf.fit(X_train,np.ravel(Y_train))

# Make predictions on the testing data
Y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(Y_test, Y_pred)

# Print the classification report
print('Classification report for Random Forest:')
print('')
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test, Y_pred ))
print(f'Accuracy: {accuracy:.2f}')

# ------------------------------------------------------------------------------
# SVM
# ------------------------------------------------------------------------------
# Create an SVM classifier with a linear or non-linear kernel (e.g., 'linear', 'rbf', 'poly', 'sigmoid')
clf = SVC(kernel='linear', C=1, random_state=42)

# Train the classifier on the training data
clf.fit(X_train,np.ravel(Y_train))

# Make predictions on the testing data
Y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(Y_test, Y_pred)

# Print the classification report
print('Classification report for SVM:')
print('')
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test, Y_pred ))
print(f'Accuracy: {accuracy:.2f}')

# Create a Gradient Boosting classifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train the classifier on the training data
clf.fit(X_train,np.ravel(Y_train))

# Make predictions on the testing data
Y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(Y_test, Y_pred)

# Print the classification report
print('Classification report for Gradient Boosting Classifier:')
print('')
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test, Y_pred ))
print(f'Accuracy: {accuracy:.2f}')