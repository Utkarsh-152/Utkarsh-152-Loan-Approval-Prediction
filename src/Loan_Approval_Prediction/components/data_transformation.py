import pandas as pd
import numpy as np
from src.Loan_Approval_Prediction.logger import logging
from src.Loan_Approval_Prediction.exception import CustomException
import sys
import os
from src.Loan_Approval_Prediction import utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    transformed_data_path: str = os.path.join("artifacts", "transformed_data")
    X_train_data_path: str = os.path.join("artifacts", "transformed_data", "X_train.csv")
    X_valid_data_path: str = os.path.join("artifacts", "transformed_data", "X_valid.csv")
    X_test_data_path: str = os.path.join("artifacts", "transformed_data", "X_test.csv")
    y_train_data_path: str = os.path.join("artifacts", "transformed_data", "y_train.csv")
    y_valid_data_path: str = os.path.join("artifacts", "transformed_data", "y_valid.csv")
    y_test_data_path: str = os.path.join("artifacts", "transformed_data", "y_test.csv")
    preprocessor_path: str = os.path.join("artifacts", "transformed_data", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            logging.info("Creating StandardScaler object")
            scaler = StandardScaler()
            return scaler
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            # Create transformed_data directory if it doesn't exist
            os.makedirs(self.data_transformation_config.transformed_data_path, exist_ok=True)
            
            # Read the train and test data
            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            
            logging.info("Performing feature engineering on train data")
            # Convert categorical variables to dummy variables for train data
            train_dummies = pd.get_dummies(train_df, dtype='int')
            train_dummies = train_dummies.drop(['education_Not Graduate', 'self_employed_No', 'loan_status_Rejected'], axis=1)
            train_dummies.rename(columns={
                'education_Graduate': 'education',
                'self_employed_Yes': 'self_employed',
                'loan_status_Approved': 'loan_status'
            }, inplace=True)
            
            logging.info("Performing feature engineering on test data")
            # Apply same transformations to test data
            test_dummies = pd.get_dummies(test_df, dtype='int')
            test_dummies = test_dummies.drop(['education_Not Graduate', 'self_employed_No', 'loan_status_Rejected'], axis=1)
            test_dummies.rename(columns={
                'education_Graduate': 'education',
                'self_employed_Yes': 'self_employed',
                'loan_status_Approved': 'loan_status'
            }, inplace=True)
            
            logging.info("Splitting data into features and target")
            y_train = train_dummies['loan_status']
            X_train_full = train_dummies.drop(['loan_status'], axis=1)
            
            y_test = test_dummies['loan_status']
            X_test = test_dummies.drop(['loan_status'], axis=1)
            
            logging.info("Splitting train data into train and validation sets")
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train, test_size=0.25, random_state=42
            )
            
            logging.info("Applying StandardScaler transformation")
            preprocessor = self.get_data_transformer_object()
            preprocessor.fit(X_train)
            
            X_train_scaled = preprocessor.transform(X_train)
            X_val_scaled = preprocessor.transform(X_val)
            X_test_scaled = preprocessor.transform(X_test)
            
            logging.info("Saving transformed data")
            # Convert scaled arrays back to dataframes with column names
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=X_val.columns)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            
            # Save transformed datasets
            X_train_scaled_df.to_csv(self.data_transformation_config.X_train_data_path, index=False)
            X_val_scaled_df.to_csv(self.data_transformation_config.X_valid_data_path, index=False)
            X_test_scaled_df.to_csv(self.data_transformation_config.X_test_data_path, index=False)
            
            pd.DataFrame(y_train).to_csv(self.data_transformation_config.y_train_data_path, index=False)
            pd.DataFrame(y_val).to_csv(self.data_transformation_config.y_valid_data_path, index=False)
            pd.DataFrame(y_test).to_csv(self.data_transformation_config.y_test_data_path, index=False)
            
            # Save the preprocessor
            utils.save_object(
                file_path=self.data_transformation_config.preprocessor_path,
                obj=preprocessor
            )
            
            logging.info("Data transformation completed successfully")
            
            return (
                X_train_scaled_df, X_val_scaled_df, X_test_scaled_df,
                y_train, y_val, y_test,
                self.data_transformation_config.preprocessor_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)



