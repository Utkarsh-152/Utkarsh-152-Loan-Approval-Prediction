from src.Loan_Approval_Prediction.logger import logging
from src.Loan_Approval_Prediction.exception import CustomException
import sys
import os
import pandas as pd
from dataclasses import dataclass
from src.Loan_Approval_Prediction.utils import get_data_from_mysql
from sklearn.model_selection import train_test_split

logging.info("Data Ingestion libraries imported successfully")

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts","raw.csv")
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        
        try:
            data = get_data_from_mysql()
            logging.info("Reading completed from mysql database")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,header=True,index=False)
            logging.info("Data saved successfully to csv")

            train_set,test_set = train_test_split(data,test_size=0.2,random_state=42)
            train_set.drop(columns=['loan_id'],inplace=True)
            test_set.drop(columns=['loan_id'],inplace=True)
            train_set.to_csv(self.ingestion_config.train_data_path,header=True,index=False)
            test_set.to_csv(self.ingestion_config.test_data_path,header=True,index=False)
            logging.info("Data split successfully into train and test")


            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Error occured while reading dataset")
            raise CustomException(e, sys)



