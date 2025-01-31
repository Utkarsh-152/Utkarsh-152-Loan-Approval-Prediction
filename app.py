from src.Loan_Approval_Prediction.logger import logging
from src.Loan_Approval_Prediction.exception import CustomException
from src.Loan_Approval_Prediction.components.data_ingestion import DataIngestion
from src.Loan_Approval_Prediction.components.data_ingestion import DataIngestionConfig
from src.Loan_Approval_Prediction.components.data_transformation import DataTransformation
from src.Loan_Approval_Prediction.components.data_transformation import DataTransformationConfig
from src.Loan_Approval_Prediction.components.model_trainer import ModelTrainer
from src.Loan_Approval_Prediction.components.model_trainer import ModelTrainerConfig
from src.Loan_Approval_Prediction.components.model_performace_monitering import ModelPerformanceMonitoring
from src.Loan_Approval_Prediction.pipelines.prediction_pipeline import PredictPipeline, CustomData
import sys
import pandas as pd
from flask import Flask, request, jsonify, render_template
import os
import csv
from datetime import datetime


if __name__ == "__main__":

    logging.info("Application has started")

    try:
        data_ingestion = DataIngestion()
        data_ingestion_config = DataIngestionConfig()
        train_data,test_data = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed successfully")

        data_transformation = DataTransformation()
        data_transformation_config = DataTransformationConfig()
        X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, y_train, y_val, y_test, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)
        logging.info("Data transformation completed successfully")

        model_trainer = ModelTrainer()
        model_trainer_config = ModelTrainerConfig()
        model_trainer.initiate_model_trainer(X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, y_train, y_val, y_test)
        best_model = model_trainer.initiate_model_trainer(X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, y_train, y_val, y_test)
        print(f"Best model found on both training and validation set is {best_model}")
        logging.info("Model training completed successfully")

        monitor = ModelPerformanceMonitoring()

        metrics = monitor.generate_report(best_model, X_test_scaled_df, y_test)
        logging.info("Model performance monitoring completed successfully")
    except Exception as e:
        logging.info("Error occured while running the application")
        raise CustomException(e, sys)



app = Flask(__name__)

@app.route('/', methods=['GET'])
def render_index_page():
    return render_template('index.html', debug=True)

@app.route('/predict', methods=['GET', 'POST'])
def get_data():
    if request.method == 'GET':
        return render_template('data_collection.html', debug=True)
    else:
        try:
            # Get form data and log it
            data = request.get_json()
            logging.info(f"Received data: {data}")

            # Create CustomData object
            custom_data = CustomData(
                no_of_dependents=int(data['no_of_dependents']),
                education=data['education'],
                self_employed=data['self_employed'],
                income_annum=float(data['income_annum']),
                loan_amount=float(data['loan_amount']),
                loan_term=float(data['loan_term']),
                cibil_score=float(data['cibil_score']),
                residential_assets_value=float(data['residential_assets_value']),
                commercial_assets_value=float(data['commercial_assets_value']),
                luxury_assets_value=float(data['luxury_assets_value']),
                bank_asset_value=float(data['bank_asset_value'])
            )

            # Get data as DataFrame
            input_df = custom_data.get_data_as_dataframe()
            logging.info("DataFrame created successfully")

            # Initialize prediction pipeline
            predict_pipeline = PredictPipeline()
            prediction = predict_pipeline.predict(input_df)[0]
            prediction_result = 'approved' if prediction == 1 else 'rejected'
            logging.info(f"Prediction made: {prediction_result}")
            
            return jsonify({
                'prediction': int(prediction),
                'message': prediction_result
            })

        except Exception as e:
            logging.error(f"Error processing data: {str(e)}")
            return jsonify({'error': str(e)}), 500

@app.route('/result', methods=['GET'])
def show_result():
    result = request.args.get('result', '')
    return render_template('result.html', result=result)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_df = pd.DataFrame([data])
        prediction = PredictPipeline().predict(input_df)[0]
        prediction_result = 'approved' if prediction == 1 else 'rejected'
        

        return jsonify({
            'prediction': int(prediction),
            'message': prediction_result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
