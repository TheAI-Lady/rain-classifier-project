# Import argument parser
import argparse

#Import modules
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Scalers
from sklearn.preprocessing import StandardScaler

# Models for classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

# Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class Engineering:
    """
    Methods/tools for cleaning, pre-processing and feature engineering the data
    """
    def __init__(self):
        pass
    
    def duplicate_rows(self, data):
        """
        Check and remove duplicate rows in the dataframe. 
        """
        duplicate_rows = data[data.duplicated()]
        print(duplicate_rows)
        data = data.drop_duplicates()
        return data
    
    def identify_features(self, data):
        """
         Identifies numerical, discrete, and continuous features in the given DataFrame.
         Returns lists of column names for each type of feature.
        """
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
 
        return num_feats, discrete_feats, continuous_feats
    
    def handle_invalid_data(self, data):
        """
        # Check for invalid data for categorical and discrete column
        # To display the labels in each categorical and discrete columns
        """
        column_names = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm']

        for column in column_names:
            categories = set(filter(lambda x: not np.isnan(x) if isinstance(x, float) else True, data[column]))
            print(f"Categories for {column}:")
            for category in categories:
                print(f" - {category}")
            print()


    def handle_uppercase(self, data):
        """
        Change Pressure Data columns To Uppercase for consistency
        """
        data['Pressure9am'] = data['Pressure9am'].str.upper()
        data['Pressure3pm'] = data['Pressure3pm'].str.upper()
        return data
       
        
    def non_numerical_data_check(self, data):
        """
        # To check for non-numerical data in each numerical-continuous columns
        """
        column_names = ['Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'AverageTemp']

        def is_non_numerical(value):
            return not isinstance(value, (int, float))

        for column in column_names:
          non_numerical_data = list(filter(is_non_numerical, data[column]))
        print(f"Non-numerical data for {column}:")
        for value in non_numerical_data:
            print(f" - {value}")
        print()

    
    def negative_to_absolute(self, data):
        """
        # Convert the negative value in the Sunshine column to positive
        """
        data['Sunshine'] = data['Sunshine'].abs()
        return data
        
    def missing_data(self, data):
        """
        # Count missing data for each column
        """
        missing_data_count = data.isnull().sum()
    
    def plot_data(self, data, discrete_feats, continuous_feats):
        """
        # Create bar plot for each discrete feature
        # Create histogram for each continuous feature
        # Get summary statistics for each numerical column
        """
        for feat in discrete_feats:
            plt.figure()
            data[feat].value_counts().plot(kind='bar')
            plt.title(feat)
            plt.xlabel('Value')
            plt.ylabel('Count')

        plt.show()

        for feat in continuous_feats:
            plt.figure()
            plt.hist(data[feat], bins=20)
            plt.title(feat)
            plt.xlabel('Value')
            plt.ylabel('Count')
        plt.show()

    def summary_statistics(self, data):
        """
        # Get summary statistics for each numerical column
        """
        summary_statistics = data.describe()

    def handle_mean_median_mode(self, data):
        """
        Fill missing field with mean, median or mode accordingly to histogram visualisation. 
        """
        # Calculate the mean of Humidity3pm and fill missing data with the mean value.
        col_mean = int(data['Humidity3pm'].mean())
        data['Humidity3pm'].fillna(col_mean, inplace=True)

        # Calculate the mean of Humidity9am and fill missing data with the mean value.
        col_mean = int(data['Humidity9am'].mean())
        data['Humidity9am'].fillna(col_mean, inplace=True)

        # List of specified column names containing continuous numerical data with missing values.
        column_names = [
            'WindGustSpeed', 'WindSpeed3pm', 'Sunshine', 'Evaporation', 'Humidity3pm', 'Rainfall', 'Humidity9am', 'WindSpeed9am', 'AverageTemp']

        # Calculate the mean and fill missing values for each column
        for column in column_names:
            if data[column].dtype in ['int64', 'float64']:  # Check if the column contains numerical data
                col_mean = round(data[column].mean(), 1)  # Calculate the mean and round to one decimal place
                data[column].fillna(col_mean, inplace=True)
        
        # List of specified column names containing discrete numerical and categorical data with missing values.
        column_names = [
            'WindGustDir', 'WindDir3pm', 'WindDir9am', 'Pressure3pm', 'Pressure9am', 'Cloud3pm', 'Cloud9am']

        # Calculate the mean and fill missing values for each column
        for column in column_names:
            col_mode = data[column].mode()
            data[column].fillna(col_mode, inplace=True)

        return data


    def handle_outliers(self, data):
        """
        Determine the number of outliers.
        """
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
                outliers_count = count_outliers(data[column])
                outliers_counts[column] = outliers_count

        # Print the number of outliers for each column
        for column, count in outliers_counts.items():
            print(f"{column}: {count} outliers")



    def correlation_analysis(self, data):
        """
        To identify correlated numerical column
        """
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
        return corr_matrix
    
    def feature_engineer(self, data):
        """
        Converts the cleaned dataset to a feature engineered dataset.
        """
        # Data preparation
        X = data.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]].values
        Y = data.iloc[:, 18].values

        Y = Y.reshape(-1, 1)  # 1d list into 2d list

        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        X = imputer.fit_transform(X)
        Y = imputer.fit_transform(Y)

        # Encoding categorical features
        le1 = LabelEncoder()
        X[:, 0] = le1.fit_transform(X[:, 0])
        le2 = LabelEncoder()
        X[:, 4] = le2.fit_transform(X[:, 4])
        le3 = LabelEncoder()
        X[:, 6] = le3.fit_transform(X[:, 6])
        le4 = LabelEncoder()
        X[:, 7] = le4.fit_transform(X[:, 7])
        le5 = LabelEncoder()
        X[:, 12] = le5.fit_transform(X[:, 12])
        le6 = LabelEncoder()
        X[:, 13] = le6.fit_transform(X[:, 13])
        le7 = LabelEncoder()
        Y[:, 0] = le7.fit_transform(Y[:, 0])

        Y = np.array(Y, dtype=float)

        # Feature scaling
        sc = StandardScaler()
        X = sc.fit_transform(X)

        return X, Y
       
    def check_imbalance(self, data):
        # Count 'yes' and 'no' occurrences for dependent parameter
        counts = data['RainTomorrow'].value_counts()
        print("RainTomorrow :\n", counts)

    def handle_imbalance(self, X, Y):
        # Handle imbalanced Data
        oversample = SMOTE()
        X, Y = oversample.fit_resample(X, Y)
        return X, Y
     
    
class ML_Pipeline(Engineering):
    """
    Machine learning pipeline for the AIAP14 technical assessment.
    Arguments:
    scaler: Scaler class to be used. Default scaler=StandardScaler() 
    model_selection: Model class to be used. 
        See ML_Pipeline().classification_models for available classification models.
    """
    def __init__(self, model_selection=None, scaler=StandardScaler()):
        self.scaler = scaler
        self.model_selection = model_selection
        self.is_injested = False
        self.is_pre_processed = False
        self.is_split = False
        self.is_fitted = False
        self.is_predict = False
        self.data_raw = None
        self.data_processed = None
        self.features = None
        self.target = None
        
        self.scalers = {"StandardScaler":StandardScaler(),}
               
        self.classification_models = [LogisticRegression(),                                    
                                     RandomForestClassifier(),
                                     SVC(),
                                     GradientBoostingClassifier()]
        
        self.classification_model_names = [type(x).__name__ for x in self.classification_models]
        
        self.metrics = ['accuracy_score', 
                        'classification_report']
        
        if type(self.model_selection).__name__ in self.classification_model_names:
            self.model_type = 'Classification'        
    
    def settings(self):
        """
        Prints the current settings.
        """
        print("Model selection:", type(self.model_selection).__name__)
        print("Model type:     ", self.model_type)
        print("Scaler type:    ", type(self.scaler).__name__)
        if self.model_type=='Classification':
            print("Metric:          Recall/Accuracy")
        print()
    
    def injest_data(self, from_path="../data/fishing.db", verbose=False):
        """
        Injests the given data
        Arguments:
        from_path: path to database file
        verbose: If True, print status.
        """
        if self.is_injested:
            print("Data has already been injested.")
        else:
            conn = sqlite3.connect(from_path)
            self.data_raw = pd.read_sql_query(sql="SELECT * FROM fishing", con=conn)
            
            self.is_injested = True
            
            if verbose:
                print("Imported data of shape {}".format(self.data_raw.shape))

 
    def pre_process(self, verbose=False):
        """
        Pre-process the given data
        Arguments:
        verbose: If True, print status.
        """
        if not self.is_injested:
            print("Data has not been injested.")
        elif self.is_pre_processed:
            print("Data has already been pre-processed.")
        else:
            data = self.data_raw
           
            # Handle and drop duplicate rows
            data = self.duplicate_rows(data)

            # Identifies numerical, discrete, and continuous features in the given DataFrame.
            num_feats, discrete_feats, continuous_feats = self.identify_features(data)

            # Handle invalid data
            # self.handle_invalid_data(data)
            data = self.handle_uppercase(data)
            # self.non_numerical_data_check(data)
            data = self.negative_to_absolute(data)

            # Handle missing data
            self.missing_data(data)
            # self.plot_data(data, discrete_feats, continuous_feats)
            # self.summary_statistics(data)
            data = self.handle_mean_median_mode(data)  #fill missing data with mean, median, mode

            #check number of outliers
            self.handle_outliers(data)
          
            # Feature engineer a processed dataframe, one hot encoding for discrete and categorical columns, and data scaling
            X, Y = self.feature_engineer(data)

            self.check_imbalance(data)
            X, Y = self.handle_imbalance(X, Y)            
            
            self.data_processed = data
            self.features = X
            
            # if self.model_type=='Classification':
            #     self.target = self.data_processed.rain_tomorrow.values

            if self.model_type=='Classification':
                self.target = Y

            self.is_pre_processed = True
                        
            if verbose:
                print("Pre-processed data to shape {}.".format(self.data_processed.shape), end=' ')
                _removed = len(self.data_raw)-len(self.data_processed)
                print("{} duplicated/incomplete entries removed.".format(_removed))


    def split(self, test_size=0.25, random_state=42, verbose=False):
        """
        Splits the given data into train and test sets
        Arguments:
        test_size: Float, should be between 0.0 and 1.0 and represent the 
            proportion of the dataset to include in the test split.
        random_state: int, RandomState
        verbose: If True, print status.
        """
        if not self.is_injested:
            print("Data has not been injested.")
        elif not self.is_pre_processed:
            print("Data has not been pre-processed")
        elif self.is_split:
            print("Data has already been split into train/test")
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.features, self.target, test_size=test_size, random_state=random_state)
            self.is_split = True
            
            if verbose:
                print("Train test split complete. {} train size & {} test size."\
                      .format(len(self.y_train),len(self.y_test)))
    
    def fit(self, verbose=False):
        """
        Fits the given training data.
        Arguments:
        verbose: If True, print status.
        """
        if not self.is_injested:
            print("Data has not been injested.")
        elif not self.is_pre_processed:
            print("Data has not been pre-processed")
        elif not self.is_split:
            print("Data has not been split into train/test")
        elif self.is_fitted:
            print("Model has already been fitted.")
        else:
            self.pipe = Pipeline([('scaler', self.scaler), ('model', self.model_selection)])
            self.pipe.fit(self.x_train, self.y_train)
            
            self.is_fitted = True
            
            if verbose:
                print("Model has been fitted.")
        
    def predict(self, x_test=None, verbose=False):
        """
        Predicts the targets for the test features.
        Arguments:
        X_test: 2D-array, If None, predicts for the original test dataset. 
            Else, predicts the provided dataset.
        verbose: If True, print status.
        """
        if not self.is_injested:
            print("Data has not been injested.")
        elif not self.is_pre_processed:
            print("Data has not been pre-processed")
        elif not self.is_split:
            print("Data has not been split into train/test")
        elif not self.is_fitted:
            print("Model has not been fitted.")
        else:
            if x_test is None:
                self.y_pred = self.pipe.predict(self.x_test)
            else:
                self.y_pred = self.pipe.predict(x_test)
            
            self.is_predict = True
            
            if verbose:
                print("Prediction complete. Evaluation can be made.")
        
    def evaluate(self, y_test=None):
        """
        Evaluates the model based on actual targets and predictions.
        Arguments:
        y_test: 2D-array, If None, evaluates for the original test dataset. 
            Else, evaluates the provided dataset.
        """
        if not self.is_predict:
            print("No prediction has been made yet.")
        else:
            if y_test is None:
                _y_test = self.y_test
            else:
                _y_test = y_test
                
            if self.model_type=='Classification':
                print(classification_report(_y_test, self.y_pred))
                


























