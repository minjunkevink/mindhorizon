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

### Project Structure
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
        db_actions.py
    models/
        model.joblib
        (will potential add additional models to test different architectures)
    templates/
        dashboard.html
        history.html
        home.html
        user_projection.html
    scripts/
        preprocessing.py
        train.py
    requirements.txt
    README.md
```

### Preprocess/Train/Evaluation (Important Points to Note)

*Preprocessing*:
- Used one hot encoding to convert 'yes' and 'no' to 1s and 0s
- Split the train-test ratio as 80-20

*Training*:
- Used a simple linear regression model (TODO: try other architectures)
- Saved the model as a `.joblib` under `models/`

*Evaluation*:
- LRmodels shows:
    - `MSE: 4.0826283985, R^2 score: 0.9889832909573145`
    - With our data range being [10,100] MSE is sufficiently small
    - Our model explains 98% of the variance in our target which is very sufficient

### Functionality

MindHorizon offers the following key features:

*   **User Authentication:** Secure user accounts with login and logout functionality.
*   **Performance Prediction:** Predicts student performance based on user-provided metrics.
*   **Personalized Projections:** Generates personalized performance projections for the upcoming week based on user input and historical data.
*   **Data Visualization:**
    *   Visualizes the distribution of the original dataset's performance index, along with the normal average and projected average performance of all users.
    *   Provides a visualization of a user's projected performance over the next 7 days, showing potential improvement with consistent effort.
*   **Historical Data Tracking:** Allows users to view their historical performance metrics.

### Usage

1.  **Register or Login:** Create an account or log in with your existing credentials.
2.  **Dashboard:** After logging in, you'll be redirected to the dashboard where you can:
    *   Enter your study metrics (hours studied, previous scores, sleep hours, etc.).
    *   View your predicted performance based on the entered metrics.
3.  **Generate Projection:** Click the "Generate Projection" button to create a personalized performance projection for the next 7 days.
4.  **View Projection:** The projection will be displayed on a separate page accessible through a personalized URL.
5.  **History:** View your historical metrics data.
