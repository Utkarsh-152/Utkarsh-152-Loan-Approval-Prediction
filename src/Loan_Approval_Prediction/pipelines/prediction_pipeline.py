import sys
import os
import pandas as pd
import numpy as np
from src.Loan_Approval_Prediction.exception import CustomException
from src.Loan_Approval_Prediction.logger import logging
from src.Loan_Approval_Prediction import utils

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "transformed_data", "preprocessor.pkl")

    def predict(self, features):
        try:
            # Load the model and preprocessor
            model = utils.load_object(self.model_path)
            preprocessor = utils.load_object(self.preprocessor_path)

            logging.info(f"Features before one hot encoding: {features.columns}")

            # Convert categorical variables to dummy variables
            data = pd.get_dummies(features, dtype='int', drop_first=False)

            if 'self_employed_No' in data.columns:
                data['self_employed'] = 0
                data.drop('self_employed_No', axis=1, inplace=True)

            if 'education_Not Graduate' in data.columns:
                data['education'] = 0
                data.drop('education_Not Graduate', axis=1, inplace=True)

            logging.info(f"Data after one hot encoding: {data.columns}")            

            # Rename columns to match training data
            data.rename(columns={
                'education_Graduate': 'education',
                'self_employed_Yes': 'self_employed'
            }, inplace=True)

            # Reorder columns to match desired order
            desired_order = [
                'no_of_dependents',
                'income_annum',
                'loan_amount',
                'loan_term',
                'cibil_score',
                'residential_assets_value',
                'commercial_assets_value',
                'luxury_assets_value',
                'bank_asset_value',
                'education',
                'self_employed'
            ]
            data = data[desired_order]

            # Scale the features using the saved preprocessor
            scaled_data = preprocessor.transform(data)
            scaled_data = pd.DataFrame(scaled_data, columns=data.columns)

            # Make prediction
            prediction = model.predict(scaled_data)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
        self,
        no_of_dependents: str,
        education: str,
        self_employed: str,
        income_annum: float,
        loan_amount: float,
        loan_term: float,
        cibil_score: float,
        residential_assets_value: float,
        commercial_assets_value: float,
        luxury_assets_value: float,
        bank_asset_value: float
    ):
        self.no_of_dependents = no_of_dependents
        self.education = education
        self.self_employed = self_employed
        self.income_annum = income_annum
        self.loan_amount = loan_amount
        self.loan_term = loan_term
        self.cibil_score = cibil_score
        self.residential_assets_value = residential_assets_value
        self.commercial_assets_value = commercial_assets_value
        self.luxury_assets_value = luxury_assets_value
        self.bank_asset_value = bank_asset_value

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "no_of_dependents": [self.no_of_dependents],
                "education": [self.education],
                "self_employed": [self.self_employed],
                "income_annum": [self.income_annum],
                "loan_amount": [self.loan_amount],
                "loan_term": [self.loan_term],
                "cibil_score": [self.cibil_score],
                "residential_assets_value": [self.residential_assets_value],
                "commercial_assets_value": [self.commercial_assets_value],
                "luxury_assets_value": [self.luxury_assets_value],
                "bank_asset_value": [self.bank_asset_value]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
