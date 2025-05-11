from werkzeug.security import generate_password_hash
from extensions import db
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import func

Base = declarative_base()


# Define the Airline model
class Airline(db.Model):
    __tablename__ = 'airlines'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, unique=True, nullable=False)

class FlightNumber(db.Model):
    __tablename__ = 'flight_numbers'

    id = db.Column(db.Integer, primary_key=True)
    flight_number = db.Column(db.String, unique=True, nullable=False)

class Airport(db.Model):
    __tablename__ = 'airports'

    id = db.Column(db.Integer, primary_key=True)
    airport_name = db.Column(db.String, nullable=False)

class AirlineFlight(db.Model):
    __tablename__ = 'airline_flights'

    id = db.Column(db.Integer, primary_key=True)
    airline_id = db.Column(db.Integer, db.ForeignKey('airlines.id'), nullable=False)
    flight_number_id = db.Column(db.Integer, db.ForeignKey('flight_numbers.id'), nullable=False)
    start_airport_id = db.Column(db.Integer, db.ForeignKey('airports.id'), nullable=False)
    end_airport_id = db.Column(db.Integer, db.ForeignKey('airports.id'), nullable=False)
    delay_probability = db.Column(db.Integer)
    day_of_week = db.Column(db.Integer)
    time = db.Column(db.Integer)
    length = db.Column(db.Integer)
    delay = db.Column(db.Boolean, nullable=True)
    start_at = db.Column(db.String, nullable=True)

    airline = db.relationship('Airline', backref=db.backref('flights', lazy=True))
    flight_number = db.relationship('FlightNumber', backref=db.backref('flights', lazy=True))
    airport_from = db.relationship('Airport', foreign_keys=[start_airport_id], backref=db.backref('departure_flights', lazy=True))
    airport_to = db.relationship('Airport', foreign_keys=[end_airport_id], backref=db.backref('arrival_flights', lazy=True))

# Define the User model
class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String, unique=True, nullable=False)
    nick = db.Column(db.String, unique=True, nullable=False)
    password = db.Column(db.String, nullable=False)
    admin = db.Column(db.Boolean, default=False)

    def __init__(self, email, nick, password, admin=False):
        self.email = email
        self.nick = nick
        self.password = generate_password_hash(password)
        self.admin = admin

class System(db.Model):
    __tablename__ = 'system'

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String, nullable=False)
    value = db.Column(db.String, nullable=False)

class GeneratedModels(db.Model):
    __tablename__ = 'generated_models'

    id = db.Column(db.Integer, primary_key=True)
    default = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime, default=func.now(), nullable=False)
    fields = db.Column(db.String, nullable=False)
    accuracy = db.Column(db.Float, nullable=True)
