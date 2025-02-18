# Loan Approval Prediction System

A sophisticated machine learning-based loan approval prediction system that leverages advanced algorithms to assess loan eligibility. Built with Flask, GSAP animations, and modern ML techniques.

## 🌟 Key Features

- Real-time loan eligibility prediction
- Interactive web interface with GSAP animations
- Comprehensive EDA (Exploratory Data Analysis)
- Model performance monitoring
- Secure data processing
- Responsive design
- RESTful API endpoints

## 🛠️ Technical Stack

### Backend
- **Framework:** Flask
- **Database:** MySQL
- **ML Libraries:** 
  - Scikit-learn
  - Pandas
  - NumPy
- **Data Processing:** StandardScaler, One-Hot Encoding

### Frontend
- HTML5/CSS3
- GSAP Animations
- Bootstrap 5
- Responsive Design

### ML Pipeline
- Data Ingestion
- Data Transformation
- Model Training
- Performance Monitoring
- Prediction Pipeline

## 📊 Model Performance

- Accuracy: ~97.4%
- Precision: ~98%
- Recall: ~98%
- F1 Score: ~98%

## 🚀 Installation & Setup

1. Clone the repository
```bash
git clone https://github.com/Utkarsh-152/Utkarsh-152-Loan-Approval-Prediction.git
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
.\venv\Scripts\activate  # For Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```
g
4. Run the application
```bash
python app.py
```

## 📁 Project Structure

```
loan-approval-prediction/
├── src/
│   └── Loan_Approval_Prediction/
│       ├── components/
│       ├── pipelines/
│       ├── utils.py
│       ├── logger.py
│       └── exception.py
├── templates/
├── static/
├── notebooks/
├── artifacts/
└── logs/
```

## 🔄 ML Pipeline Flow

1. **Data Ingestion**: Fetches data from MySQL database
2. **Data Transformation**: Handles preprocessing and feature engineering
3. **Model Training**: Trains multiple models and selects the best performer
4. **Performance Monitoring**: Tracks model metrics and generates reports
5. **Prediction Pipeline**: Handles real-time predictions

## 📝 API Documentation

### Prediction Endpoint
```http
POST /predict
Content-Type: application/json

{
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 500000,
    "loan_amount": 2000000,
    "loan_term": 15,
    "cibil_score": 750
    // ... other fields
}
```

## 👥 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 👤 Author

**Utkarsh Tripathi**
- Email: shreytripathi2004@gmail.com
- LinkedIn: [Utkarsh Tripathi](https://linkedin.com/in/utkarsh-tripathi-0144001b2/)
- GitHub: [Utkarsh-152](https://github.com/Utkarsh-152)
