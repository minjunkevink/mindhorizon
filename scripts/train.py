import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# paths to training and test data
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_path = os.path.join(project_root, "data", "train")
test_path = os.path.join(project_root, "data", "test")

# load processed data
X_train = pd.read_csv(os.path.join(train_path, "X_train_processed.csv"))
y_train = pd.read_csv(os.path.join(train_path, "y_train_processed.csv"))
X_test = pd.read_csv(os.path.join(test_path, "X_test_processed.csv"))
y_test = pd.read_csv(os.path.join(test_path, "y_test_processed.csv"))

# initialize and train 
model = LinearRegression()
model.fit(X_train, y_train)

# eval
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R^2 score:", r2)

# save the trained model to model/ directory 
models_path = os.path.join(project_root, "models")
os.makedirs(models_path, exist_ok=True)
model_filename = os.path.join(models_path, "model.joblib")
joblib.dump(model, model_filename)

print("Model saved to:", model_filename)