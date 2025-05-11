import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from models import AirlineFlight, GeneratedModels, System
from sqlalchemy import func
import pandas as pd
from extensions import db
from flask import Flask
from dotenv import load_dotenv

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.secret_key = os.getenv("SESSION_KEY")

    app.config[
        'SQLALCHEMY_DATABASE_URI'] = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    return app

# Оновлення прогресу в таблиці System
def update_progress(progress):
    app = create_app()
    with app.app_context():
        progress_percent = System.query.filter_by(key='progress_percent').first()

        if progress_percent:
            progress_percent.value = str(progress)
            db.session.commit()

def train_random_forest(data, new_model_id):
    app = create_app()
    with app.app_context():
        generated_models = GeneratedModels.query.filter_by(default=True).first()
        fields = generated_models.fields.split(", ")

        data_to_generate = []

        for item in data:
            row = {"delay": getattr(item, "delay")}
            for field in fields:
                if hasattr(item, field):
                    row[field] = getattr(item, field)
                else:
                    row[field] = None
            data_to_generate.append(row)

        df = pd.DataFrame(data_to_generate)

        X = df.drop('delay', axis=1)
        y = df['delay']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)

        update_progress(10)

        model.fit(X_train, y_train)

        update_progress(50)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        update_progress(90)

        model_save_path = f"images/{new_model_id}/random_forest_model.pkl"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(model, model_save_path)

        model_entry = GeneratedModels.query.filter_by(id=new_model_id).first()
        model_entry.accuracy = accuracy
        db.session.commit()

        update_progress(100)

        in_progress = System.query.filter_by(key='in_progress').first()
        in_progress.value = "false"
        db.session.commit()

    return model

def random_forest_worker(task_queue, props):
    app = create_app()
    with app.app_context():
        while True:
            task = task_queue.get()

            if task == "EXIT":
                print("Worker thread finished.")
                break

            if task == "Train Random Forest Model":

                data = AirlineFlight.query.filter(
                    AirlineFlight.airline_id.isnot(None),
                    AirlineFlight.flight_number_id.isnot(None),
                    AirlineFlight.start_airport_id.isnot(None),
                    AirlineFlight.end_airport_id.isnot(None),
                    AirlineFlight.day_of_week.isnot(None),
                    AirlineFlight.time.isnot(None),
                    AirlineFlight.length.isnot(None),
                    AirlineFlight.delay.isnot(None)
                ).order_by(func.random()).all()

                train_random_forest(data, props['new_model_id'])
