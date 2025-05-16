from flask import Flask, render_template, redirect, url_for, send_from_directory, request, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import os
from dotenv import load_dotenv
from wtforms import PasswordField, BooleanField, StringField, SelectField, IntegerField
from wtforms.validators import InputRequired, Email, Length, EqualTo
from flask_wtf import FlaskForm

from models import User, Airline, FlightNumber, Airport, AirlineFlight, System, GeneratedModels
from extensions import db
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import func
from scipy.stats import chi2_contingency
import json
from sklearn.preprocessing import MinMaxScaler
import matplotlib.patches as mpatches
import joblib
import numpy as np

import threading
import queue
from random_forest import random_forest_worker

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SESSION_KEY")

app.config[
    'SQLALCHEMY_DATABASE_URI'] = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)


class LoginForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email()], default='')
    password = PasswordField('Password', validators=[InputRequired()], default='')


class RegistrationForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email()], default='')
    nick = StringField('nick', validators=[InputRequired(), Length(min=4, max=20)], default='')
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8)], default='')
    confirm_password = PasswordField('Confirm Password', validators=[InputRequired(), EqualTo('password')], default='')


class ResetPasswordRequestForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email()], default='')


class EditUserForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email()], default='')
    nick = StringField('nick', validators=[InputRequired(), Length(min=4, max=20)], default='')
    id = IntegerField('id', validators=[InputRequired()])
    admin = BooleanField('admin', default=False)


class CreateUserForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email()], default='')
    nick = StringField('nick', validators=[InputRequired(), Length(min=4, max=20)], default='')
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8)], default='')
    id = IntegerField('id', validators=[InputRequired()])
    admin = BooleanField('admin', default=False)


class CreateAirlineForm(FlaskForm):
    airline = StringField('Airline', validators=[InputRequired()], default='')
    flight_number = StringField('Flight', validators=[InputRequired()], default='')
    airport_from = StringField('Airport From', validators=[InputRequired()], default='')
    airport_to = StringField('Airport To', validators=[InputRequired()], default='')
    day_of_week = StringField('Day Of Week', validators=[InputRequired()], default='')
    time = StringField('Time', validators=[InputRequired()], default='')
    start_at = StringField('Start At ', validators=[InputRequired()], default='')
    length = StringField('Length', validators=[InputRequired()], default='')
    delay = SelectField('Delay', choices=[(None, 'Flight not completed yet'), (True, 'On Time'), (False, 'Delayed')],
                        coerce=str, default=None)


class EditAirlineForm(FlaskForm):
    airline = StringField('Airline', validators=[InputRequired()], default='')
    flight_number = StringField('Flight', validators=[InputRequired()], default='')
    airport_from = StringField('Airport From', validators=[InputRequired()], default='')
    airport_to = StringField('Airport To', validators=[InputRequired()], default='')
    day_of_week = StringField('Day Of Week', validators=[InputRequired()], default='')
    time = StringField('Time', validators=[InputRequired()], default='')
    start_at = StringField('Start At ', validators=[InputRequired()], default='')
    length = StringField('Length', validators=[InputRequired()], default='')
    id = IntegerField('id', validators=[InputRequired()])
    delay = SelectField('Delay', choices=[(None, 'Flight not completed yet'), (True, 'On Time'), (False, 'Delayed')],
                        coerce=str, default=None)


@app.route('/')
def index():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        per_page = request.args.get('per_page', 25, type=int)
        page = request.args.get('page', 1, type=int)
        items = AirlineFlight.query.filter(AirlineFlight.start_at.isnot(None)).order_by(
            AirlineFlight.id.desc()).paginate(page=page,
                                              per_page=per_page,
                                              error_out=True)
        pages = list(range(max(1, items.page - 2), min(items.pages, items.page + 2) + 1))

        return render_template('index.html', items=items, user=user, pages=pages)
    return redirect(url_for('login'))


@app.route('/images/<path:filename>')
def get_image(filename):
    try:
        base_directory = 'images'

        if os.path.exists(os.path.join(base_directory, filename)):
            return send_from_directory(base_directory, filename)
    except FileNotFoundError:
        return jsonify({"error": "Image not found"}), 404


@app.route('/users')
def users():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        per_page = request.args.get('per_page', 25, type=int)
        page = request.args.get('page', 1, type=int)
        items = User.query.order_by(User.id.desc()).paginate(page=page, per_page=per_page, error_out=True)
        pages = list(range(max(1, items.page - 2), min(items.pages, items.page + 2) + 1))

        return render_template('users.html', items=items, user=user, pages=pages)
    return redirect(url_for('login'))


@app.route('/airlines')
def airlines():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        per_page = request.args.get('per_page', 25, type=int)
        page = request.args.get('page', 1, type=int)
        items = AirlineFlight.query.order_by(AirlineFlight.id.desc()).paginate(page=page, per_page=per_page,
                                                                               error_out=True)
        pages = list(range(max(1, items.page - 2), min(items.pages, items.page + 2) + 1))

        return render_template('airlines.html', items=items, user=user, pages=pages)
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            return redirect(url_for('index'))
        else:
            flash('Incorrect username or password!')

    form = LoginForm()
    return render_template('login.html', form=form)


@app.route('/registration', methods=['GET', 'POST'])
def registration():
    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']

        if User.query.filter_by(nick=username).first():
            flash('Username already exists! Please choose a different one.')
        else:
            hashed_password = generate_password_hash(password)

            new_user = User(email=email, nick=username, password=hashed_password, admin=False)
            db.session.add(new_user)
            db.session.commit()

            flash('Registration successful! You can now log in.')
            return redirect(url_for('login'))

    form = RegistrationForm()
    return render_template('registration.html', form=form)


@app.route('/forgot-password', methods=['GET'])
def forgotPassword():
    form = ResetPasswordRequestForm()
    return render_template('forgot-password.html', form=form)


@app.route('/logout', methods=['GET'])
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))


@app.route('/create-user', methods=['GET', 'POST'])
def create_user():
    if request.method == 'POST':
        email = request.form['email']
        nick = request.form['nick']
        password = request.form['password']
        admin = request.form.get('admin') == 'True'
        hashed_password = generate_password_hash(password)

        new_user = User(email=email, nick=nick, password=hashed_password, admin=admin)

        db.session.add(new_user)
        db.session.commit()
        flash("User created successfully!", "success")
        return redirect(url_for('users'))

    form = CreateUserForm()
    return render_template('create-user.html', form=form)


@app.route('/edit-user/<int:id>', methods=['GET', 'POST'])
def edit_user(id):
    user = User.query.get_or_404(id)

    if request.method == 'POST':
        user.email = request.form['email']
        user.nick = request.form['nick']
        user.admin = request.form.get('admin') == 'True'

        db.session.commit()
        flash("User updated successfully!", "success")
        return redirect(url_for('users'))

    form = EditUserForm(obj=user)
    return render_template('edit-user.html', user=user, form=form)


@app.route('/delete-user', methods=['POST'])
def delete_user():
    user_id = request.form['id']
    user_to_delete = User.query.get(user_id)

    if user_to_delete:
        db.session.delete(user_to_delete)
        db.session.commit()
        flash("User deleted successfully!", "success")
    else:
        flash("User not found!", "danger")

    return redirect(url_for('users'))


@app.route('/create-airline', methods=['GET', 'POST'])
def create_airline():
    if request.method == 'POST':
        airline_name = request.form['airline']
        flight_number = request.form['flight_number']
        airport_from = request.form['airport_from']
        airport_to = request.form['airport_to']
        day_of_week = request.form['day_of_week']
        length = request.form['length']
        delay = request.form['delay']
        start_at = request.form['start_at']
        time = request.form['time']

        if delay == '':
            delay = None
        else:
            delay = delay == 'True'

        airline = Airline.query.filter_by(name=airline_name).first()
        if not airline:
            airline = Airline(name=airline_name)
            db.session.add(airline)
            db.session.commit()

        flight = FlightNumber.query.filter_by(flight_number=flight_number).first()
        if not flight:
            flight = FlightNumber(flight_number=flight_number)
            db.session.add(flight)
            db.session.commit()

        start_airport = Airport.query.filter_by(airport_name=airport_from).first()
        if not start_airport:
            start_airport = Airport(airport_name=airport_from)
            db.session.add(start_airport)
            db.session.commit()

        end_airport = Airport.query.filter_by(airport_name=airport_to).first()
        if not end_airport:
            end_airport = Airport(airport_name=airport_to)
            db.session.add(end_airport)
            db.session.commit()

        prediction = get_delay_probability({
            "airline_id": airline.id,
            "flight_number_id": flight.id,
            "start_airport_id": start_airport.id,
            "end_airport_id": end_airport.id,
            "day_of_week": int(day_of_week),
            "length": int(length),
            "time": int(time),
        })

        new_flight = AirlineFlight(
            airline_id=airline.id,
            flight_number_id=flight.id,
            start_airport_id=start_airport.id,
            end_airport_id=end_airport.id,
            delay_probability=prediction.probability_of_delay,
            day_of_week=int(day_of_week) if day_of_week else None,
            length=int(length) if length else None,
            delay=delay,
            time=int(time) if time else None,
            start_at=start_at
        )

        db.session.add(new_flight)
        db.session.commit()

        flash("AirlineFlight created successfully!", "success")
        return redirect(url_for('airlines'))

    form = CreateAirlineForm()
    return render_template('create-airline.html', form=form)


@app.route('/edit-airline/<int:id>', methods=['GET', 'POST'])
def edit_airline(id):
    airline_flight = AirlineFlight.query.get_or_404(id)

    if request.method == 'POST':
        airline_name = request.form['airline']
        flight_number = request.form['flight_number']
        airport_from = request.form['airport_from']
        airport_to = request.form['airport_to']
        day_of_week = request.form['day_of_week']
        start_at = request.form['start_at']
        time = request.form['time']
        length = request.form['length']

        airline = Airline.query.filter_by(name=airline_name).first()
        if not airline:
            airline = Airline(name=airline_name)
            db.session.add(airline)
            db.session.commit()

        flight = FlightNumber.query.filter_by(flight_number=flight_number).first()
        if not flight:
            flight = FlightNumber(flight_number=flight_number)
            db.session.add(flight)
            db.session.commit()

        start_airport = Airport.query.filter_by(airport_name=airport_from).first()
        if not start_airport:
            start_airport = Airport(airport_name=airport_from)
            db.session.add(start_airport)
            db.session.commit()

        end_airport = Airport.query.filter_by(airport_name=airport_to).first()
        if not end_airport:
            end_airport = Airport(airport_name=airport_to)
            db.session.add(end_airport)
            db.session.commit()

        airline_flight.day_of_week = int(day_of_week) if day_of_week else None
        airline_flight.start_at = start_at
        airline_flight.time = int(time) if time else None
        airline_flight.length = int(length) if length else None

        delay_value = request.form.get('delay')
        if delay_value == '':
            airline_flight.delay = None
        else:
            airline_flight.delay = delay_value == 'True'

        airline_flight.airline_id = airline.id
        airline_flight.flight_number_id = flight.id
        airline_flight.start_airport_id = start_airport.id
        airline_flight.end_airport_id = end_airport.id

        prediction = get_delay_probability({
            "airline_id": airline.id,
            "flight_number_id": flight.id,
            "start_airport_id": start_airport.id,
            "end_airport_id": end_airport.id,
            "day_of_week": int(day_of_week),
            "length": int(length),
            "time": int(time),
        })

        airline_flight.delay_probability = prediction.probability_of_delay

        db.session.commit()

        flash("AirlineFlight updated successfully!", "success")
        return redirect(url_for('airlines'))

    form = EditAirlineForm(airline=airline_flight.airline.name,
                           flight_number=airline_flight.flight_number.flight_number,
                           airport_from=airline_flight.airport_from.airport_name,
                           airport_to=airline_flight.airport_to.airport_name,
                           day_of_week=airline_flight.day_of_week,
                           time=airline_flight.time,
                           start_at=airline_flight.start_at,
                           length=airline_flight.length,
                           id=airline_flight.id,
                           delay=airline_flight.delay)
    return render_template('edit-airline.html', airline=airline_flight, form=form)


@app.route('/delete-airlines', methods=['POST'])
def delete_airlines():
    airline_id = request.form['id']
    airline_to_delete = AirlineFlight.query.get(airline_id)

    if airline_to_delete:
        db.session.delete(airline_to_delete)
        db.session.commit()
        flash("AirlineFlight deleted successfully!", "success")
    else:
        flash("AirlineFlight not found!", "danger")

    return redirect(url_for('airlines'))


def get_system_statuses():
    in_progress = System.query.filter_by(key='in_progress').first()
    progress_percent = System.query.filter_by(key='progress_percent').first()

    in_progress_value = in_progress.value if in_progress else 'false'
    progress_percent_value = progress_percent.value if progress_percent else '0'

    return {
        'in_progress': in_progress_value,
        'progress_percent': progress_percent_value
    }


@app.route('/system', methods=['GET'])
def system():
    per_page = request.args.get('per_page', 25, type=int)
    page = request.args.get('page', 1, type=int)
    items = GeneratedModels.query.order_by(GeneratedModels.id.desc()).paginate(page=page, per_page=per_page,
                                                                               error_out=True)
    pages = list(range(max(1, items.page - 2), min(items.pages, items.page + 2) + 1))
    status_details = get_system_statuses()
    return render_template('system.html', in_progress=status_details['in_progress'],
                           progress_percent=status_details['progress_percent'], items=items, pages=pages)


@app.route('/model-update-status', methods=['GET'])
def get_model_update_status():
    return jsonify(get_system_statuses())


@app.route('/check-model', methods=['POST'])
def check_model():
    GeneratedModels.query.update({GeneratedModels.default: False})

    model = GeneratedModels.query.get(request.form['id'])
    model.default = True
    db.session.commit()
    return jsonify({'message': 'Default model updated successfully!'}), 200


def model_dependencies():
    folder = 'images/create';

    if os.path.exists(folder):
        file_path = os.path.join(folder, "dependencies_results.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                dependencies = json.load(file)
            return dependencies

    os.makedirs(folder)

    airlines_data = AirlineFlight.query.filter(
        AirlineFlight.airline_id.isnot(None),
        AirlineFlight.flight_number_id.isnot(None),
        AirlineFlight.start_airport_id.isnot(None),
        AirlineFlight.end_airport_id.isnot(None),
        AirlineFlight.day_of_week.isnot(None),
        AirlineFlight.time.isnot(None),
        AirlineFlight.length.isnot(None),
        AirlineFlight.delay.isnot(None)
    ).order_by(func.random()).all()

    data = pd.DataFrame([{
        'airline_id': item.airline_id,
        'flight_number_id': item.flight_number_id,
        'start_airport_id': item.start_airport_id,
        'end_airport_id': item.end_airport_id,
        'day_of_week': item.day_of_week,
        'time': item.time,
        'length': item.length,
        'delay': item.delay
    } for item in airlines_data])

    numerical_fields = ['time', 'length']
    categorical_fields = ['flight_number_id', 'airline_id', 'start_airport_id', 'end_airport_id', 'day_of_week']

    scaler = MinMaxScaler()

    for field in numerical_fields:
        data[field] = scaler.fit_transform(data[[field]])
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=data['delay'], y=data[field], palette='Set2')
        plt.title(f"Relation between {field} and Delay")
        plt.xlabel('Delay (1 = Yes, 0 = No)')
        plt.ylabel(field)
        plt.savefig(f'{folder}/{field}_vs_delay.png')
        plt.close()

    for field in categorical_fields:
        category_delay_freq = data.groupby([field, 'delay']).size().unstack(fill_value=0)
        category_delay_freq = category_delay_freq.div(category_delay_freq.sum(axis=1),
                                                      axis=0)

        plt.figure(figsize=(10, 6))

        category_delay_freq.plot(kind='bar', stacked=True, colormap='Set2')

        no_delay_patch = mpatches.Patch(color='#66c2a5', label='No Delay (0)')
        delay_patch = mpatches.Patch(color='#fc8d62', label='Delay (1)')
        plt.legend(handles=[no_delay_patch, delay_patch], title='Delay', loc='upper right')

        plt.title(f"Frequency of Delay by {field}")
        plt.xlabel(field)
        plt.ylabel('Frequency of Delay')
        plt.xticks(rotation=45)

        plt.savefig(f'{folder}/{field}_vs_delay.png')
        plt.close()

    correlation_matrix = data[numerical_fields + ['delay']].corr()

    chi2_results = {}

    for field in categorical_fields:
        contingency_table = pd.crosstab(data[field], data['delay'])
        # Perform the chi-square test
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        chi2_results[field] = p

    dependencies = {
        'correlation_dict': {field: correlation_matrix[field]['delay'] for field in numerical_fields},
        'chi2_dict': chi2_results
    }

    output_file = f"{folder}/dependencies_results.json"
    with open(output_file, 'w') as json_file:
        json.dump(dependencies, json_file, indent=4)

    return dependencies


@app.route('/create-new-model', methods=['GET'])
def create_new_model():
    dependencies = model_dependencies()

    correlation_dict = dependencies['correlation_dict']
    chi2_dict = dependencies['chi2_dict']

    # Поріг кореляції та p-value
    correlation_threshold = 0.7
    p_value_threshold = 0.05

    recommended_fields = []

    for field, corr_value in correlation_dict.items():
        if abs(corr_value) > correlation_threshold:
            recommended_fields.append(field)

    for field, p_value in chi2_dict.items():
        if p_value < p_value_threshold:
            recommended_fields.append(field)

    if len(recommended_fields) < 3:
        for field, corr_value in sorted(correlation_dict.items(), key=lambda item: abs(item[1]), reverse=True):
            if field not in recommended_fields:
                recommended_fields.append(field)
            if len(recommended_fields) >= 3:
                break

        for field, p_value in sorted(chi2_dict.items(), key=lambda item: item[1]):
            if field not in recommended_fields:
                recommended_fields.append(field)
            if len(recommended_fields) >= 3:
                break

    return render_template('create-new-model.html', recommended_fields=recommended_fields)


@app.route('/approve-generating', methods=['POST'])
def approve_generating():
    data = request.get_json()

    fields = []

    if data.get('time'):
        fields.append('time')

    if data.get('length'):
        fields.append('length')

    if data.get('day_of_week'):
        fields.append('day_of_week')

    if data.get('flight_number_id'):
        fields.append('flight_number_id')

    if data.get('airline_id'):
        fields.append('airline_id')

    if data.get('start_airport_id'):
        fields.append('start_airport_id')

    if data.get('end_airport_id'):
        fields.append('end_airport_id')

    GeneratedModels.query.update({GeneratedModels.default: False})

    new_model = GeneratedModels(
        fields=', '.join(fields),
        default=True
    )

    db.session.add(new_model)
    db.session.commit()

    new_model_id = new_model.id

    create_folder_path = 'images/create'
    new_folder_path = f'images/{new_model_id}'

    in_progress = System.query.filter_by(key='in_progress').first()
    progress_percent = System.query.filter_by(key='progress_percent').first()

    in_progress.value = 'true'
    progress_percent.value = '0'

    db.session.commit()

    if os.path.exists(create_folder_path):
        os.rename(create_folder_path, new_folder_path)

    worker_thread = threading.Thread(target=random_forest_worker, args=(task_queue,),
                                     kwargs={'props': {"new_model_id": new_model_id}})
    worker_thread.start()

    task_queue.put("Train Random Forest Model")

    return jsonify({})

def get_delay_probability(data_to_generate):
    generated_models = GeneratedModels.query.filter_by(default=True).first()

    fields = generated_models.fields.split(", ")

    features = []
    for field in fields:
        features.append(data_to_generate[field])

    features = np.array(features).reshape(1, -1)

    model_path = f"images/{generated_models.id}/random_forest_model.pkl"

    if not os.path.exists(model_path):
        return {'prediction': 0, 'probability_of_delay': 0}

    model = joblib.load(model_path)

    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)
    delay_probability = prediction_proba[0][1]
    delay_probability = round(float(delay_probability * 100), 2)

    if delay_probability > 99:
        delay_probability = 99

    if delay_probability == 0:
        delay_probability = 1

    return {'prediction': int(prediction[0]), 'probability_of_delay': delay_probability}

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()

    airline_name = input_data['airline']
    flight_number = input_data['flight_number']
    airport_from = input_data['airport_from']
    airport_to = input_data['airport_to']

    airline = Airline.query.filter_by(name=airline_name).first()
    if not airline:
        airline = Airline(name=airline_name)
        db.session.add(airline)
        db.session.commit()

    flight = FlightNumber.query.filter_by(flight_number=flight_number).first()
    if not flight:
        flight = FlightNumber(flight_number=flight_number)
        db.session.add(flight)
        db.session.commit()

    start_airport = Airport.query.filter_by(airport_name=airport_from).first()
    if not start_airport:
        start_airport = Airport(airport_name=airport_from)
        db.session.add(start_airport)
        db.session.commit()

    end_airport = Airport.query.filter_by(airport_name=airport_to).first()
    if not end_airport:
        end_airport = Airport(airport_name=airport_to)
        db.session.add(end_airport)
        db.session.commit()

    prediction = get_delay_probability({
        "airline_id": airline.id,
        "flight_number_id": flight.id,
        "start_airport_id": start_airport.id,
        "end_airport_id": end_airport.id,
        "day_of_week": int(input_data['day_of_week']),
        "length": int(input_data['length']),
        "time": int(input_data['time']),
    })

    return jsonify(prediction)


if __name__ == '__main__':
    task_queue = queue.Queue()
    app.run(debug=True)
