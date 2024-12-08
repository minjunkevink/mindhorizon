## MindHorizon 
final project for applied python fall'24

### Project Overview

Mind Horizon is a habit-tracking and performance prediction application. The goal of the project is to allow users to log daily metrics related to their study habits, sleep, and extracurricular activities, and then use a trained machine learning model to predict their future performance index. Ultimately, the application will provide insights and suggestions to help users improve their habits and academic outcomes.

### Dataset Description

We are using a Student Performance dataset, which contains 10,000 records with the following features:
- Hours Studied (numeric)
- Previous Scores (numeric)
- Extracurricular Activities (categorical: “Yes”/“No”)
- Sleep Hours (numeric)
- Sample Question Papers Practiced (numeric)
- Performance Index (target variable, numeric)

The target variable, Performance Index, ranges from 10 to 100 and represents a measure of academic performance.

Project Structure
```
mindhorizon/
    data/
        Student_Performance.csv
        train/
            X_train_processed.csv
            y_train_processed.csv
        test/
            X_test_processed.csv
            y_test_processed.csv
    src/
        app.py
    models/
        model.joblib
        (will potential add additional models to test different architectures)
    scripts/
        preprocessing.py
        train.py
    requirements.txt
    README.md
```

