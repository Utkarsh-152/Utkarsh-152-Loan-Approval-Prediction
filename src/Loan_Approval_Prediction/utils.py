import os
import sys
import logging
from src.Loan_Approval_Prediction.logger import logging
from src.Loan_Approval_Prediction.exception import CustomException
import pymysql
import pandas as pd
from dotenv import load_dotenv  
from sqlalchemy import create_engine
import pickle
from sklearn.metrics import accuracy_score
load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")


def get_data_from_mysql():

    try:

        logging.info("Data readfrom mysql started")
        mydb = pymysql.connect(host=host,user=user,password=password,db=db)
        engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{db}")  # Example for MySQL

        data = pd.read_sql_query("SELECT * FROM loan_approval_dataset",engine)
        logging.info("Data read successfully from mysql")
        return data
    
    except Exception as e:
        logging.info("Error occured while reading data from mysql")
        raise CustomException(e, sys)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, models):
    try:

        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]    
            model.fit(X_train, y_train)

            y_val_pred = model.predict(X_val)
            
            val_model_score = accuracy_score(y_val, y_val_pred)

            report[list(models.keys())[i]] = val_model_score
        return report
    
    except Exception as e:
        logging.info("Error occured while evaluating models")
        raise CustomException(e, sys)


