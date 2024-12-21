# pip install scikit-learn==1.3.2
# pip install numpy
# pip install flask


# load packages==============================================================
from flask import Flask, jsonify, render_template, request, session, redirect, url_for
import mysql.connector
import pickle
import numpy as np

import plotly
import plotly.graph_objects as go
import json

app = Flask(__name__)


# Load the scaler, label encoder, model, and class names=====================
scaler = pickle.load(open("Models\\scaler.pkl",'rb'))
model = pickle.load(open("Models\\model.pkl", 'rb'))



class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Social Network Studies',
               'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
               'Banker', 'Writer', 'Accountant', 'Designer',
               'Construction Engineer', 'Game Developer', 'Stock Investor',
               'Real Estate Developer']

# Recommendations ===========================================================
def Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                    weekly_self_study_hours, math_score, history_score, physics_score,
                    chemistry_score, biology_score, english_score, geography_score,
                    total_score, average_score):
    # Encode categorical variables
    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    extracurricular_activities_encoded = 1 if extracurricular_activities else 0

    # Create feature array
    feature_array = np.array([[gender_encoded, part_time_job_encoded, absence_days, extracurricular_activities_encoded,
                               weekly_self_study_hours, math_score, history_score, physics_score,
                               chemistry_score, biology_score, english_score, geography_score, total_score,
                               average_score]])

    # Scale features
    scaled_features = scaler.transform(feature_array)

    # Predict using the model
    probabilities = model.predict_proba(scaled_features)

    # Get top five predicted classes along with their probabilities
    top_classes_idx = np.argsort(-probabilities[0])[:5]
    top_classes_names_probs = [(class_names[idx], probabilities[0][idx]) for idx in top_classes_idx]

    return top_classes_names_probs


app.secret_key = "xtay6UY&"
# MySQL configurations
mysql_config = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "ers",
}

@app.route("/")
def home():
    if "name" in session:
        return render_template("home.html")
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        print("username: ",username)

        conn = mysql.connector.connect(**mysql_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM user WHERE username = %s AND password = %s", (username, password)
        )
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        

        if user:
            session["name"] = user["name"]
            session["username"] = user["username"]
            session['user_id'] = user['id']
            return redirect(url_for("dashboard"))
        else:
            return render_template(
                "login.html", error="Sorry Invalid username or password", username=username
            )
    
    
    return render_template("login.html")

# logout
@app.route('/logout')
def logout():
    # Remove the user's information from the session
    session.pop('name', None)
    session.pop('username', None)
    
    # Redirect the user to the login page
    return redirect(url_for('login'))

# Signup
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        username = request.form["username"]
        password = request.form["password"]

        if name == "" or username == "" or password == "":
            return render_template(
                "signup.html", error="Sorry!. fields must be filled"
            )
        if len(password) < 6:
            return render_template(
                "signup.html", error="Sorry! Password must be at least 6 characters."
            )

        # Check if email already exists
        conn = mysql.connector.connect(**mysql_config)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user WHERE username = %s", (username,))
        user = cursor.fetchone()
        if user:
            cursor.close()
            conn.close()
            return render_template(
                "signup.html", error="Sorry!. his username already exists", username = username
            )

        # Insert user into database
        cursor.execute(
            cursor.execute(
                "INSERT INTO user (name, username, password) VALUES (%s, %s, %s)",
                (name, username, password),
            ),
            (name, username, password),
        )
        conn.commit()
        cursor.close()
        conn.close()

        # session['username'] = name
        return redirect(url_for("login"))
    return render_template("signup.html")


# @app.route('/dashboard')
# def dashboard():
#     conn = mysql.connector.connect(**mysql_config)
#     cursor = conn.cursor()

#     # Get the total number of predictions
#     cursor.execute("SELECT COUNT(*) AS total_predictions FROM recommendations")
#     result = cursor.fetchone()
#     total_predictions = result[0]

#     cursor.execute("SELECT * FROM recommendations ORDER BY id DESC LIMIT 5")
#     last_predictions = cursor.fetchall()

#     # Fetch all the records from the recommendations table
#     cursor.execute("SELECT * FROM recommendations ORDER BY id DESC LIMIT 5")
#     all_predictions = cursor.fetchall()

#     # Get the logged-in user's information
#     user_id = session.get('user_id')
#     cursor.execute("SELECT * FROM user WHERE id = %s", (user_id,))
#     user_data = cursor.fetchone()

#     cursor.execute("SELECT COUNT(*) AS total_users FROM user")
#     result = cursor.fetchone()
#     total_users = result[0]


#     cursor.close()
#     conn.close()

#     return render_template('dashboard.html',
#                        total_predictions=total_predictions,
#                        last_predictions=last_predictions,
#                        user_data=user_data,
#                        total_users=total_users,
#                        all_predictions=all_predictions)


@app.route('/dashboard', methods=['GET', 'POST'])
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    conn = mysql.connector.connect(**mysql_config)
    cursor = conn.cursor()

    # Get the total number of predictions
    cursor.execute("SELECT COUNT(*) AS total_predictions FROM recommendations")
    result = cursor.fetchone()
    total_predictions = result[0]

    # Handle custom date range filter
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')

    # Create query based on custom date range
    if start_date and end_date:
        cursor.execute("SELECT * FROM recommendations WHERE DATE(system_date) BETWEEN %s AND %s ORDER BY id DESC", (start_date, end_date))
    else:
        cursor.execute("SELECT * FROM recommendations ORDER BY id DESC LIMIT 5")  # Default to the last 5 predictions if no date range is provided
    
    all_predictions = cursor.fetchall()

    # Get the logged-in user's information
    user_id = session.get('user_id')
    cursor.execute("SELECT * FROM user WHERE id = %s", (user_id,))
    user_data = cursor.fetchone()

    cursor.execute("SELECT COUNT(*) AS total_users FROM user")
    result = cursor.fetchone()
    total_users = result[0]

    cursor.close()
    conn.close()

    return render_template('dashboard.html',
                           total_predictions=total_predictions,
                           user_data=user_data,
                           total_users=total_users,
                           all_predictions=all_predictions,
                           start_date=start_date,
                           end_date=end_date)




# @app.route("/pred", methods=["POST"])
# def pred():
#     # Retrieve the form data
#     gender = request.form['gender']
#     part_time_job = request.form['part_time_job']
#     absence_days = int(request.form['absence_days'])
#     extracurricular_activities = request.form['extracurricular_activities']
#     weekly_self_study_hours = int(request.form['weekly_self_study_hours'])
#     math_score = int(request.form['math_score'])
#     history_score = int(request.form['history_score'])
#     physics_score = int(request.form['physics_score'])
#     chemistry_score = int(request.form['chemistry_score'])
#     biology_score = int(request.form['biology_score'])
#     english_score = int(request.form['english_score'])
#     geography_score = int(request.form['geography_score'])
#     total_score = float(request.form['total_score'])
#     average_score = float(request.form['average_score'])
#     username = request.form['username']

#      # Call the function to get the recommendations
#     recommendations = Recommendations(
#         gender,
#         part_time_job,
#         absence_days,
#         extracurricular_activities,
#         weekly_self_study_hours,
#         math_score,
#         history_score,
#         physics_score,
#         chemistry_score,
#         biology_score,
#         english_score,
#         geography_score,
#         total_score,
#         average_score,
#     )
    

#     conn = mysql.connector.connect(**mysql_config)
#     cursor = conn.cursor()

#     recommendation_1 = recommendations[0][0]
#     recommendation_1_value = recommendations[0][1]*100

#     # Insert the form data into the database
#     sql = "INSERT INTO recommendations (gender, part_time_job, absence_days, extracurricular_activities, weekly_self_study_hours, math_score, history_score, physics_score, chemistry_score, biology_score, english_score, geography_score, total_score, average_score, username,predicted_class,probability) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
#     values = (gender, part_time_job, absence_days, extracurricular_activities, weekly_self_study_hours, math_score, history_score, physics_score, chemistry_score, biology_score, english_score, geography_score, total_score, average_score, username, recommendation_1,recommendation_1_value)
#     cursor.execute(sql, values)
#     conn.commit()

#     # Close the database connection
#     conn.close()

#     # Prepare the data to be returned
#     data = {
#         'username': username,
#         'gender': gender,
#         'part_time_job': part_time_job,
#         'absence_days': absence_days,
#         'extracurricular_activities': extracurricular_activities,
#         'weekly_self_study_hours': weekly_self_study_hours,
#         'math_score': math_score,
#         'history_score': history_score,
#         'physics_score': physics_score,
#         'chemistry_score': chemistry_score,
#         'biology_score': biology_score,
#         'english_score': english_score,
#         'geography_score': geography_score,
#         'total_score': total_score,
#         'average_score': average_score
#     }
    

#     # Combine the form data and recommendations into a single response
#     response = {
#         'form_data': data,
#         'recommendations': recommendations
#     }

    

#     return jsonify(response)
@app.route("/pred", methods=["POST"])
def pred():
    # Retrieve the form data
    gender = request.form['gender']
    part_time_job = request.form['part_time_job']
    absence_days = int(request.form['absence_days'])
    extracurricular_activities = request.form['extracurricular_activities']
    weekly_self_study_hours = int(request.form['weekly_self_study_hours'])
    math_score = int(request.form['math_score'])
    history_score = int(request.form['history_score'])
    physics_score = int(request.form['physics_score'])
    chemistry_score = int(request.form['chemistry_score'])
    biology_score = int(request.form['biology_score'])
    english_score = int(request.form['english_score'])
    geography_score = int(request.form['geography_score'])
    total_score = float(request.form['total_score'])
    average_score = float(request.form['average_score'])
    username = request.form['username']

    # Call the function to get the recommendations
    recommendations = Recommendations(
        gender,
        part_time_job,
        absence_days,
        extracurricular_activities,
        weekly_self_study_hours,
        math_score,
        history_score,
        physics_score,
        chemistry_score,
        biology_score,
        english_score,
        geography_score,
        total_score,
        average_score,
    )
    
    conn = mysql.connector.connect(**mysql_config)
    cursor = conn.cursor()

    recommendation_1 = recommendations[0][0]
    recommendation_1_value = recommendations[0][1]*100

    # Insert the form data into the database
    sql = """
    INSERT INTO recommendations 
    (gender, part_time_job, absence_days, extracurricular_activities, 
    weekly_self_study_hours, math_score, history_score, physics_score, 
    chemistry_score, biology_score, english_score, geography_score, 
    total_score, average_score, username, predicted_class, probability) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (
        gender, part_time_job, absence_days, extracurricular_activities, 
        weekly_self_study_hours, math_score, history_score, physics_score, 
        chemistry_score, biology_score, english_score, geography_score, 
        total_score, average_score, username, recommendation_1, recommendation_1_value
    )
    cursor.execute(sql, values)
    conn.commit()

    # Update total predictions and users
    cursor.execute("SELECT COUNT(*) FROM recommendations")
    total_predictions = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM user")
    total_users = cursor.fetchone()[0]


    conn.close()

    # Combine the form data, recommendations, and updated dashboard info into a single response
    response = {
        'form_data': {
            'username': username,
            'gender': gender,
            'part_time_job': part_time_job,
            'absence_days': absence_days,
            'extracurricular_activities': extracurricular_activities,
            'weekly_self_study_hours': weekly_self_study_hours,
            'math_score': math_score,
            'history_score': history_score,
            'physics_score': physics_score,
            'chemistry_score': chemistry_score,
            'biology_score': biology_score,
            'english_score': english_score,
            'geography_score': geography_score,
            'total_score': total_score,
            'average_score': average_score,
        },
        'recommendations': recommendations,
        'total_predictions': total_predictions,
        'total_users': total_users,
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, port=5007)
