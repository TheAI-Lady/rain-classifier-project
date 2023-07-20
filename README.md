**dawn_lim_515E**
Full Name: Dawn Lim
Email: thediningnook@gmail.com

# Submission Folder Structure
├── src
│ ├── main.py
│ └── model_evaluation.py
├── README.md
├── eda.ipynb
├── requirements.txt
└── run.sh



# Instructions for the "run.sh" File

The default command for the shell script is as follows:
$ python ./src/main.py --db_path ./data/fishing.db

**TO SELECT MODEL PARAMETERS:**
usage: main.py [-h] [--db_path DB_PATH]
                          

Settings
optional arguments:
  -h, --help            show this help message and exit
  --db_path DB_PATH     Database directory: default = './data/fishing.db'
 
NOTE: Default filepath is relative to this the execution of the run.sh directory, edit this command to specify another path or an absolute path.

# Flow of Pipeline
1 - Injest Data: Setup connection to database and query for data
2 - Pre-processing:
     - Remove duplicate rows.
     - Check for invalid and erroneous data:
		- Change pressure data label to uppercase.	
		- Handle Sunshine Negative Hours.
     - Handle missing data.
		- Identify with column has missing data.
		- Plot the distribution curve for the numerical data.
     	- Replace missing numerical data with mean or median depending on the distribution curve.
		- Replace missing categorical data with mode.
	- Handle Outliers
      - Correlation Analysis
	- Drop irrelevant and redundant rows.
3 - One hot encoding for categorical columns.
4 - Feature scaling using scikit_learn StandardScaler.
5 - Handle imbalanced data.
6 - Split Data - This process splits the data into its required train and test set. 
7 - Fit Data - Fits the training data using the all/selected machine learning model. 
8 - Predict - Makes a prediction on the test data.
9 - Evaluate - Evaluates and print results.(Confusion Matrix, Accuracy and classification report.)

# Overview of Key Findings from EDA
- There are 1182 duplicate rows.
- Columns [Pressure9am, Pressure3pm] have labels that consist of both upper and lower case. Resolved it by converting all to uppercase.
- There are negative hours in [Sunshine]. Convert the negative values to absolute. 
- Columns with missing data[except RainToday] are less than 3%. As the number is not significant, it can be resolved by filling them with mean, median and mode.
- For the [RainToday] column, it can be removed as it is redundant information since it can be defined by RainFall( >1.0mm is defined has rain else no rain.)
- There are outliers in the numerical columns but as the total number of outliers (about 30%) is significant, thus decided not to remove them.
- Drop the [Date] and [ ColorOfBoat] columns as they are not relevant. Drop the [RainToday] column as it is redundant for the above-mentioned reason. Drop [Cloud3pm] as it has a strong 
  negative correlation of -0.7 with [Sunshine] column.
- The dependent parameter [ RainTomorrow ] is not balanced. It has 9080 'No' and 2735 'Yes'. Use SMOTE() to do oversampling.


# Model Evaluation
**Classification, Metrics & Results**
	- Recall value is the proportion of true positive predictions among all actual positive instances (all rainy days). 
	- A high recall means the model is capturing most of the rainy days, even if it also produces some false alarms.
	- Overall accuracy is also used as a secondary metric for model selection. A higher accuracy is an indication of a better-fit model. 

- A total of 4 classification models were tested, their corresponding recall values and accuracy are:
    - Logistic Regression Classifier - 79% / 80%
    - Random Forest Classifier - 92% / 90%
    - Support Vector Machine Classifier - 80% / 80%
	-Gradient Boosting Classifier - 88% / 88%

# Model Choice
**Random Forest Classifier(RFC)**
- Our goal is to determine whether it will rain tomorrow. Missing rainy days (not predicting rain when it actually rains) is more problematic, thus should focus on recall value.
- RFC model has a high recall value of 92% for predicting rain tomorrow which means that the model is capturing most of the rainy days, even if it also produces some false alarms.
- This is the recommended choice of model to be used for deployment as it has the highest recall value of 92% as well as the highest overall accuracy of 90%.


