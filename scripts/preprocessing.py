import pandas as pd
from sklearn.model_selection import train_test_split
import os

# get current directory (i assume that the csv file is in the same directory as the script)
# TODO: add that to readme
current_dir = os.getcwd()
csv_file = os.path.join(current_dir, 'Student_Performance.csv')

# read the csv file
df = pd.read_csv(csv_file)

print(df.head())
print(df.info())
print(df.describe())

# since the extracurricular activities is a categorical variable, we need to convert it to numerical (yes:1, no:0)
df['Extracurricular_Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
print(df.head())

# select features and target. in my project i will use performance index as the evalutaion metric
X = df[["Hours Studied", "Previous Scores", "Sleep Hours", 
        "Sample Question Papers Practiced", "Extracurricular_Activities"]]
y = df["Performance Index"]

# split it into train and test data and save it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.to_csv(os.path.join(current_dir, "train", "X_train_processed.csv"), index=False)
y_train.to_csv(os.path.join(current_dir, "train", "y_train_processed.csv"), index=False)
X_test.to_csv(os.path.join(current_dir, "test", "X_test_processed.csv"), index=False)
y_test.to_csv(os.path.join(current_dir, "test", "y_test_processed.csv"), index=False)