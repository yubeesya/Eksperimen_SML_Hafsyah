import pandas as pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
import joblib

def preprocessing_pipeline(csv_path):
    df = pd.read_csv(csv_path)

    # 1. Split data menjadi train dan test
    train_df, test_df = train_test_split(df, test_size=0.05, random_state=42, shuffle=True)
    train_df.resdet_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # 2. Undersampling untuk data imbalance
    df_1 = train_df[(train_df.Weather_Type == "Cloudy")]
    df_2 = train_df[(train_df.Weather_Type == "Rainy")]
    df_3 = train_df[(train_df.Weather_Type == "Sunny")]
    df_4 = train_df[(train_df.Weather_Type == "Snowy")]

    df_1_undersampled = resample(df_1, n_samples=3117, random_state=42)
    df_2_undersampled = resample(df_2, n_samples=3117, random_state=42)
    df_3_undersampled = resample(df_3, n_samples=3117, random_state=42)

    undersampled_train_df = pd.concat([df_4, df_1_undersampled, df_2_undersampled, df_3_undersampled]).reset_index(drop=True)
    undersampled_train_df = shuffle(undersampled_train_df, random_state=42)

    X_train = undersampled_train_df.drop(columns="Weather_Type", axis=1)
    y_train = undersampled_train_df["Weather_Type"]

    X_test = test_df.drop(columns="Weather_Type", axis=1)
    y_test = test_df["Weather_Type"]
