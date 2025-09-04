# Auto-generated from your MLPipelines.ipynb
# This file contains only the functions needed by the Flower integration.
# You can edit/expand it if any helper imports are missing.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from flwr.common import Context, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
import os
from sklearn.metrics import mean_squared_error
import io
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from flwr.client import NumPyClient
# Add any other imports your functions rely on below




def load_data():
    df_train = pd.read_csv('/Users/iliya/fraud/fraud/archive/fraudTrain.csv') 
    df_test = pd.read_csv('/Users/iliya/fraud/fraud/archive/fraudTest.csv')
    return df_train, df_test    


def check_for_balance(df_train):
    print("\nClass distribution in train set:")
    print(df_train['is_fraud'].value_counts())

    non_fraud_num = df_train['is_fraud'].value_counts()[0]
    fraud_num = df_train['is_fraud'].value_counts()[1]
    print(f"Number of non-fraud transactions: {non_fraud_num}")
    print(f"Number of fraud transactions: {fraud_num}") 
    print(non_fraud_num / fraud_num)

    sns.countplot(x='is_fraud', data=df_train)
    plt.title('Class Distribution in Train Set')
    plt.show()

    return non_fraud_num, fraud_num


def take_sample(df_train):
    X = df_train.drop(columns=['is_fraud'])
    y = df_train['is_fraud']    
    X, _, y, _ = train_test_split(X, y, train_size=500_000, random_state=42, stratify=y)
    df_train = pd.concat([X, y], axis=1)
    print("Sampled train set shape:", df_train.shape)
    return df_train


def feature_engineering(df_train, df_test):
    # Convert
    def compute_distance(row):
        user_loc = (row['lat'], row['long'])
        merch_loc = (row['merch_lat'], row['merch_long'])
        return geodesic(user_loc, merch_loc).km

    df_train['distance_from_home'] = df_train.apply(compute_distance, axis=1)
    df_test['distance_from_home'] = df_test.apply(compute_distance, axis=1)

    df_train['dob'] = pd.to_datetime(df_train['dob'], format='%Y-%m-%d')
    df_train['age'] = (pd.to_datetime('today') - df_train['dob']).dt.days // 365

    df_test['dob'] = pd.to_datetime(df_test['dob'], format='%Y-%m-%d')
    df_test['age'] = (pd.to_datetime('today') - df_test['dob']).dt.days // 365


    # add column time difference between last transaction and current transaction

    # For train set
    df_train['trans_date_trans_time'] = pd.to_datetime(df_train['trans_date_trans_time'])
    df_train.sort_values(['cc_num', 'trans_date_trans_time'], inplace=True)
    df_train['time_diff'] = df_train.groupby('cc_num')['trans_date_trans_time'].diff().dt.total_seconds().fillna(0)

    # For test set
    df_test['trans_date_trans_time'] = pd.to_datetime(df_test['trans_date_trans_time'])
    df_test.sort_values(['cc_num', 'trans_date_trans_time'], inplace=True)
    df_test['time_diff'] = df_test.groupby('cc_num')['trans_date_trans_time'].diff().dt.total_seconds().fillna(0)

    df_train['trans_date_trans_time'] = pd.to_datetime(df_train['trans_date_trans_time'])

    # Extract time-based features
    df_train['hour'] = df_train['trans_date_trans_time'].dt.hour         # time of day
    df_train['day'] = df_train['trans_date_trans_time'].dt.day           # day of the month
    df_train['weekday'] = df_train['trans_date_trans_time'].dt.weekday   # 0 = Monday, 6 = Sunday
    df_train['month'] = df_train['trans_date_trans_time'].dt.month
    # is weekend or no
    df_train['is_weekend'] = df_train['weekday'].apply(lambda x: 1 if x >= 5 else 0)  # 1 for Saturday/Sunday, 0 for other days
    df_train.drop(columns=['trans_date_trans_time'], inplace=True)

    df_test['trans_date_trans_time'] = pd.to_datetime(df_test['trans_date_trans_time'])

    # Extract time-based features
    df_test['hour'] = df_test['trans_date_trans_time'].dt.hour         # time of day
    df_test['day'] = df_test['trans_date_trans_time'].dt.day           # day of the month
    df_test['weekday'] = df_test['trans_date_trans_time'].dt.weekday   # 0 = Monday, 6 = Sunday
    # is weekend or no
    df_test['is_weekend'] = df_test['weekday'].apply(lambda x: 1 if x >= 5 else 0)  # 1 for Saturday/Sunday, 0 for other days
    df_test['month'] = df_test['trans_date_trans_time'].dt.month

    df_test.drop(columns=['trans_date_trans_time'], inplace=True)

    list_of_seasons = []
    for month in df_train['month']:
        if month>=1 and month<=3:
            list_of_seasons.append(1)
        elif month>=4 and month<=6:
            list_of_seasons.append(2)
        elif month>=7 and month<=9:
            list_of_seasons.append(3)
        else:
            list_of_seasons.append(4)
    df_train['season'] = list_of_seasons


    list_of_seasons_test = []
    for month in df_test['month']:
        if month>=1 and month<=3:
            list_of_seasons_test.append(1)
        elif month>=4 and month<=6:
            list_of_seasons_test.append(2)
        elif month>=7 and month<=9:
            list_of_seasons_test.append(3)
        else:
            list_of_seasons_test.append(4)
    df_test['season'] = list_of_seasons_test


    # add a column with the number of transactions per user
    df_train['num_transactions'] = df_train.groupby('cc_num')['cc_num'].transform('count')
    df_test['num_transactions'] = df_test.groupby('cc_num')['cc_num'].transform('count')


    df_train['total_amount'] = df_train.groupby('cc_num')['amt'].transform('sum')
    df_test['total_amount'] = df_test.groupby('cc_num')['amt'].transform('sum')


    df_train['avg_amt'] = df_train.groupby('cc_num')['amt'].transform('mean')
    df_test['avg_amt'] = df_test.groupby('cc_num')['amt'].transform('mean') 

    df_train['amt_std'] = df_train.groupby('cc_num')['amt'].transform('std').replace(0, 1)
    df_train['amt_z_score'] = (df_train['amt'] - df_train['avg_amt']) / df_train['amt_std']
    df_train.dropna(subset=['amt_std', 'amt_z_score'], inplace=True)

    df_test['amt_std'] = df_test.groupby('cc_num')['amt'].transform('std').replace(0, 1)
    df_test['amt_z_score'] = (df_test['amt'] - df_test['avg_amt']) / df_test['amt_std']
    df_test.dropna(subset=['amt_std', 'amt_z_score'], inplace=True)

    return df_train, df_test


def encode_categorical_features(df_train, df_test):

    # One-hot encoding for 'city', 'merchant', and 'state'
    df_train = pd.get_dummies(df_train, columns=['city', 'merchant', 'state'])
    df_test= pd.get_dummies(df_test, columns=['city', 'merchant', 'state'])

    X_train_dummies = df_train.drop(columns=['is_fraud'])
    X_test_dummies = df_test.drop(columns=['is_fraud'])
    y_train = df_train['is_fraud']
    y_test = df_test['is_fraud']
    X_train_dummies, X_test_dummies = X_train_dummies.align(X_test_dummies, join='left', axis=1, fill_value=0)

    df_train_encoded = pd.concat([X_train_dummies, y_train], axis=1)
    df_test_encoded = pd.concat([X_test_dummies, y_test], axis=1)

    # Label encoding for 'category' and 'job'
    label_cols = ['category', 'job']

    # Create a LabelEncoder instance for each column
    for col in label_cols:
        le = LabelEncoder()
        df_train_encoded[col] = le.fit_transform(df_train_encoded[col])

        # Handle unseen labels in test set
        df_test_encoded[col] = df_test_encoded[col].apply(lambda x: x if x in le.classes_ else 'unknown')

        # Add 'unknown' to classes if not present
        if 'unknown' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'unknown')

        df_test_encoded[col] = le.transform(df_test_encoded[col])

    # Binary encoding for 'gender' for train and test sets
    df_train_encoded['gender'] = df_train_encoded['gender'].map({'F': 0, 'M': 1})
    df_test_encoded['gender'] = df_test_encoded['gender'].map({'F': 0, 'M': 1})

    return df_train_encoded, df_test_encoded


def feature_selection(df_train, df_test):
    df_train.drop(columns=['Unnamed: 0','first', 'last', 'lat', 'long', 'unix_time', 'trans_num', 'street', 'dob'], inplace=True, errors='ignore')
    df_test.drop(columns=['Unnamed: 0', 'first', 'last', 'lat', 'long', 'unix_time', 'trans_num', 'street', 'dob'], inplace=True, errors='ignore')

    y_train = df_train['is_fraud']
    y_test = df_test['is_fraud']
    X_train = df_train.drop(columns=['is_fraud'])
    X_test = df_test.drop(columns=['is_fraud'])

    # feature selection using variance threshold
    # Initialize the VarianceThreshold object
    selector = VarianceThreshold(threshold=0.16)  # Set a threshold for variance
    # Fit the selector to the training data
    X_train_scaled = selector.fit_transform(X_train)
    # Transform the test data using the same selector
    X_test_scaled = selector.transform(X_test)  

    df_train_encoded = pd.DataFrame(X_train_scaled, columns=X_train.columns[selector.get_support()])
    df_test_encoded = pd.DataFrame(X_test_scaled, columns=X_test.columns[selector.get_support()])   
    df_train_encoded['is_fraud'] = y_train.values
    df_test_encoded['is_fraud'] = y_test.values

    return df_train_encoded, df_test_encoded


def resampling(df_train_encoded, df_test_encoded):

    X = df_train_encoded.drop(columns=['is_fraud'])
    y = df_train_encoded['is_fraud']
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    data_train = pd.DataFrame(X_resampled, columns=X.columns)
    data_train['is_fraud'] = y_resampled

    # Check new class distribution
    print(data_train['is_fraud'].value_counts())
    return data_train, df_test_encoded


def scale_features(data_train, df_test_encoded):
    scaler = StandardScaler()
    data_train_scaled = scaler.fit_transform(data_train.drop(columns=['is_fraud']))
    data_test_scaled = scaler.transform(df_test_encoded.drop(columns=['is_fraud']))
    data_test_scaled = pd.DataFrame(data_test_scaled, columns=df_test_encoded.columns[:-1])
    data_test_scaled['is_fraud'] = df_test_encoded['is_fraud'].values

    data_train_scaled = pd.DataFrame(data_train_scaled, columns=data_train.columns[:-1])
    data_train_scaled['is_fraud'] = data_train['is_fraud'].values
    return data_train_scaled, data_test_scaled


def split_data(data_train_scaled, data_test_scaled):
    # Split the scaled data into features and target variable
    X_train = data_train_scaled.drop(columns=['is_fraud'])
    y_train = data_train_scaled['is_fraud']
    X_test = data_test_scaled.drop(columns=['is_fraud'])
    y_test = data_test_scaled['is_fraud']

    # Return the split data
    return X_train, y_train, X_test, y_test


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model():

    xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)  # Use scale_pos_weight for imbalanced data

    return xgb_classifier
