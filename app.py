from src.Loan_Approval_Prediction.logger import logging
from src.Loan_Approval_Prediction.exception import CustomException
from src.Loan_Approval_Prediction.components.data_ingestion import DataIngestion
from src.Loan_Approval_Prediction.components.data_ingestion import DataIngestionConfig
import sys

if __name__ == "__main__":
    logging.info("Application has started")

    try:
        data_ingestion = DataIngestion()
        data_ingestion_config = DataIngestionConfig()
        train_data,test_data = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed successfully")
    except Exception as e:
        logging.info("Error occured while running the application")
        raise CustomException(e, sys)

