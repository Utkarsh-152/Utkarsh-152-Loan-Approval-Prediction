from src.Loan_Approval_Prediction.components.data_transformation import DataTransformationConfig
from src.Loan_Approval_Prediction.components.data_transformation import DataTransformation
from src.Loan_Approval_Prediction.exception import CustomException
from src.Loan_Approval_Prediction.logger import logging
import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from src.Loan_Approval_Prediction import utils

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, y_train, y_val, y_test):
        try:
            logging.info("model training started")

            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier()
            }

            model_report:dict = utils.evaluate_models(X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, y_train, y_val, y_test, models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            

            logging.info(f"Best model found on validation set is {best_model}")  


            utils.save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        
            logging.info("Model training completed successfully")

            return best_model
        
        except Exception as e:
            logging.info("Error occured while training the model")
            raise CustomException(e, sys)

















