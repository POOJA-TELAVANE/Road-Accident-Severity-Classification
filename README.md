# Road-Accident-Severity-Classification
Classification project
🧾Description: This data set is collected from Addis Ababa Sub-city police departments for master's research work. The data set has been prepared from manual records of road traffic accidents of the year 2017-20. All the sensitive information has been excluded during data encoding and finally it has 32 features and 12316 instances of the accident. Then it is preprocessed and for identification of major causes of the accident by analyzing it using different machine learning classification algorithms.



🧭 Problem Statement: The target feature is Accident_severity which is a multi-class variable. The task is to classify this variable based on the other 31 features step-by-step by going through each day's task.


Steps followed:= 
1- Importing all the necessary libraries, dataset
2- Understanding of the dataset by performing "Exploratory data analysis", Univariate, multivariate analysis of all the features
3- Analyzing duplicate values, missing values, outliers
4- Filling missing values with mode, all missing values were from categorical features
5- Perfromed feature engineering
6- Splitted dataframe into training and testing data
7- Created pipeline, applied classification based machine learning algorithms
8- Checked for accuracy and F1- weightd(target variable is highly imbalanced) for every applied algorithm
9- Found out that Random Forest algorithm worked best on this problem statement
10- Retrained random forest algorithmm after hyperparameter tuning
11- Saved the model using joblib
12- created app.py, procfile, requirements.txt, prediction file, get_model.py files for streamlit and Heroku
13- Build the app woth the help of Streamlit and deployed it as web app using Heroku



Your suggestions please!
