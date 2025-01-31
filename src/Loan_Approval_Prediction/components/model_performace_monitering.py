import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import json
from datetime import datetime

class ModelPerformanceMonitoring:
    def __init__(self):
        self.report_dir = "model_evaluation_report"
        os.makedirs(self.report_dir, exist_ok=True)

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate various performance metrics
        """
        metrics = {
            'precision': float(precision_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred)),
            'f1_score': float(f1_score(y_true, y_pred)),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        return metrics

    def generate_report(self, model, X_test, y_test):
        """
        Generate and save performance report
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred)
        
        # Add model name to metrics
        metrics['model_name'] = model.__class__.__name__
        
        # Create report with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"model_performance_report_{timestamp}.json"
        report_path = os.path.join(self.report_dir, report_name)
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Performance report saved to: {report_path}")
        return metrics




    


