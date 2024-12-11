import os
import io
import base64
import sqlite3
import secrets
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, url_for, session, flash
import joblib
from db_actions import init_db, check_credentials, insert_metrics, get_user_metrics, get_all_metrics

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'fallback_key')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, "data", "Student_Performance.csv")
model_path = os.path.join(project_root, "models", "model.joblib")

# Load dataset and model
df = pd.read_csv(data_path)
model = joblib.load(model_path)

# Compute the normal average from the dataset
normal_avg = df["Performance Index"].mean()

def compute_projected_average(model):
    """Compute the projected average performance from all user metrics in the database."""
    rows = get_all_metrics()
    if not rows:
        return None  # No user data yet, can't compute projected avg
    
    user_df = pd.DataFrame(rows, columns=[
        "Hours Studied", 
        "Previous Scores", 
        "Sleep Hours", 
        "Sample Question Papers Practiced", 
        "Extracurricular_Activities"
    ])
    
    predictions = model.predict(user_df)
    projected_avg = predictions.mean()
    return projected_avg

def create_dataset_plot(normal_avg, projected_avg=None):
    """Create a histogram of the Performance Index and optionally overlay average lines."""
    fig, ax = plt.subplots()
    df["Performance Index"].hist(bins=20, ax=ax, alpha=0.7, label='Original Dataset Distribution')
    ax.set_title("Distribution of Performance Index")
    ax.set_xlabel("Performance Index")
    ax.set_ylabel("Frequency")

    # Add line for the normal average
    ax.axvline(x=normal_avg, color='blue', linestyle='--', linewidth=2, label=f'Normal Avg: {normal_avg:.2f}')

    # If projected_avg is available, add it too
    if projected_avg is not None:
        ax.axvline(x=projected_avg, color='red', linestyle='--', linewidth=2, label=f'Projected Avg: {projected_avg:.2f}')

    ax.legend()

    pngImage = io.BytesIO()
    plt.savefig(pngImage, format='png')
    pngImage.seek(0)
    pngBase64 = base64.b64encode(pngImage.getvalue()).decode('ascii')
    plt.close(fig)
    return pngBase64

@app.route("/")
def home():
    projected_avg = compute_projected_average(model)
    plot_data = create_dataset_plot(normal_avg, projected_avg)

    if "user_id" in session:
        username = session.get("username", "User")
        return render_template("home.html", logged_in=True, username=username, plot_data=plot_data)
    else:
        return render_template("home.html", logged_in=False, plot_data=plot_data)

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")

    user_id = check_credentials(username, password)
    if user_id:
        session["user_id"] = user_id
        session["username"] = username
        flash("Logged in successfully!", "success")
        return redirect(url_for("dashboard"))
    else:
        flash("Invalid credentials. Please try again.", "error")
        return redirect(url_for("home"))

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user_id" not in session:
        flash("Please log in first.", "error")
        return redirect(url_for("home"))

    if request.method == "POST":
        try:
            hours = float(request.form.get("hours", 0))
            prev_scores = float(request.form.get("prev_scores", 0))
            sleep = float(request.form.get("sleep", 0))
            sample_qs = float(request.form.get("sample_qs", 0))
            extra = int(request.form.get("extra", 0))  # 1 for yes, 0 for no

            # Insert metrics into the database
            insert_metrics(session["user_id"], hours, prev_scores, sleep, sample_qs, extra)

            # Prepare data for prediction
            X_input = pd.DataFrame([[hours, prev_scores, sleep, sample_qs, extra]],
                                   columns=["Hours Studied", "Previous Scores", "Sleep Hours", 
                                            "Sample Question Papers Practiced", "Extracurricular_Activities"])
            prediction = model.predict(X_input)[0]

            return render_template("dashboard.html", predicted=prediction)
        except ValueError:
            flash("Please enter valid numeric values for all fields.", "error")
            return redirect(url_for("dashboard"))
    else:
        # GET request - show empty form
        return render_template("dashboard.html", predicted=None)

@app.route("/history")
def history():
    if "user_id" not in session:
        flash("Please log in first.", "error")
        return redirect(url_for("home"))

    user_metrics = get_user_metrics(session["user_id"])
    return render_template("history.html", metrics=user_metrics)

if __name__ == "__main__":
    # Ensure DB is initialized
    init_db()
    app.run(debug=True)