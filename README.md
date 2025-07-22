# Heart Failure Prediction Web Application

This project is an end-to-end machine learning web application that predicts the risk of heart disease based on clinical parameters. The application uses a trained machine learning model to provide predictions through a user-friendly web interface.

## Project Overview

The Heart Failure Prediction Web Application allows users to input their clinical data and receive a prediction about their heart disease risk. The application is built with:

- **Machine Learning Model**: Trained on the UCI Heart Disease dataset to achieve >80% accuracy
- **Flask Web Framework**: Provides the backend server and API
- **Bootstrap**: Creates a responsive and attractive user interface

## Features

- User-friendly form for entering clinical parameters
- Real-time prediction of heart disease risk
- Detailed explanation of prediction results
- Responsive design that works on desktop and mobile devices
- Information about heart disease risk factors

## Project Structure

```
/Heart Failure Prediction Web Application
│
├── model.ipynb          # Jupyter notebook with data analysis and model training
├── model.pkl            # Saved trained machine learning model
├── scaler.pkl           # Saved feature scaler for data preprocessing
├── app.py               # Flask application
├── heart.csv            # Original dataset
├── README.md            # Project documentation
│
└── /templates
    └── index.html       # HTML template for the web interface
```

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone this repository or download the project files

2. Navigate to the project directory

```bash
cd "Heart Failure Prediction Web Application"
```

3. Install the required dependencies

```bash
pip install -r requirements.txt
```

If the requirements.txt file is not available, install the following packages:

```bash
pip install flask numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Running the Application

1. To run the Jupyter Notebook for model training and analysis:

```bash
jupyter notebook model.ipynb
```

2. To start the Flask web application:

```bash
python app.py
```

3. Open your web browser and navigate to:

```
http://localhost:5000
```

## Using the Application

1. Fill in all the required fields in the form with your clinical data
2. Click the "Predict Heart Disease Risk" button
3. View your prediction result and risk assessment

## Dataset Information

The application uses the UCI Heart Disease dataset, which includes the following features:

- **Age**: Age in years
- **Sex**: Sex (1 = male, 0 = female)
- **CP**: Chest pain type (0-3)
- **Trestbps**: Resting blood pressure (mm/Hg)
- **Chol**: Serum cholesterol (mg/dl)
- **FBS**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **RestECG**: Resting electrocardiographic results (0-2)
- **Thalach**: Maximum heart rate achieved
- **Exang**: Exercise induced angina (1 = yes, 0 = no)
- **Oldpeak**: ST depression induced by exercise relative to rest
- **Slope**: Slope of the peak exercise ST segment (0-2)
- **CA**: Number of major vessels colored by fluoroscopy (0-3)
- **Thal**: Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)

## Model Performance

The machine learning model achieves over 80% accuracy on the test set. The model was trained using various classification algorithms, with the best performance achieved by a Random Forest classifier after hyperparameter tuning.

## Disclaimer

This application is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## License

This project is available for educational and personal use.

## Acknowledgments

- UCI Machine Learning Repository for providing the Heart Disease dataset
- The scikit-learn team for their machine learning library
- Flask and Bootstrap teams for their web development frameworks