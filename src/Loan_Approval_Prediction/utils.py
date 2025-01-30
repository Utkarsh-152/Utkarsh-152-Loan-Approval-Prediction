import os
import sys
import logging
from src.Loan_Approval_Prediction.logger import logging
from src.Loan_Approval_Prediction.exception import CustomException
import pymysql
import pandas as pd
from dotenv import load_dotenv  
from sqlalchemy import create_engine
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


