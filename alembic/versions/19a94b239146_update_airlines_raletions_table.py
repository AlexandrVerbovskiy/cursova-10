"""update airlines raletions table

Revision ID: 19a94b239146
Revises: 03b18df747c8
Create Date: 2025-05-10 13:54:54.087677

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from models import Airline, FlightNumber, Airport, AirlineFlight
import pandas as pd

# revision identifiers, used by Alembic.
revision: str = '19a94b239146'
down_revision: Union[str, None] = '03b18df747c8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Table creation
    op.create_table('airlines',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('name', sa.String(), nullable=False),
                    sa.PrimaryKeyConstraint('id'),
                    sa.UniqueConstraint('name')
                    )

    op.create_table('airports',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('airport_name', sa.String(), nullable=False),
                    sa.PrimaryKeyConstraint('id')
                    )

    op.create_table('flight_numbers',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('flight_number', sa.String(), nullable=False),
                    sa.PrimaryKeyConstraint('id'),
                    sa.UniqueConstraint('flight_number')
                    )

    op.create_table('airline_flights',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('airline_id', sa.Integer(), nullable=False),
                    sa.Column('flight_number_id', sa.Integer(), nullable=False),
                    sa.Column('start_airport_id', sa.Integer(), nullable=False),
                    sa.Column('end_airport_id', sa.Integer(), nullable=False),
                    sa.Column('delay_probability', sa.Integer(), nullable=True),
                    sa.Column('day_of_week', sa.Integer(), nullable=True),
                    sa.Column('time', sa.Integer(), nullable=True),
                    sa.Column('length', sa.Integer(), nullable=True),
                    sa.Column('delay', sa.Boolean(), nullable=True),
                    sa.Column('start_at', sa.VARCHAR(), nullable=True),
                    sa.ForeignKeyConstraint(['airline_id'], ['airlines.id']),
                    sa.ForeignKeyConstraint(['flight_number_id'], ['flight_numbers.id']),
                    sa.ForeignKeyConstraint(['start_airport_id'], ['airports.id']),
                    sa.ForeignKeyConstraint(['end_airport_id'], ['airports.id']),
                    sa.PrimaryKeyConstraint('id')
                    )

    # Data loading
    bind = op.get_bind()
    session = sessionmaker(bind=bind)()

    df = pd.read_csv("Airlines.csv")

    # Collection of unique values
    unique_airlines = set(df['Airline'].dropna())
    unique_flight_numbers = set(df['Flight'].dropna())
    unique_airports_from = set(df['AirportFrom'].dropna())
    unique_airports_to = set(df['AirportTo'].dropna())

    # Creating objects
    airlines = [Airline(name=name) for name in unique_airlines]
    flight_numbers = [FlightNumber(flight_number=str(num)) for num in unique_flight_numbers]
    start_airports = [Airport(airport_name=name) for name in unique_airports_from]
    end_airports = [Airport(airport_name=name) for name in unique_airports_to]

    # Adding to a session
    session.add_all(airlines)
    session.add_all(flight_numbers)
    session.add_all(start_airports)
    session.add_all(end_airports)
    session.commit()

    # Getting a mapping
    airline_id_map = {a.name: a.id for a in session.query(Airline).all()}
    flight_number_id_map = {f.flight_number: f.id for f in session.query(FlightNumber).all()}
    start_airport_id_map = {a.airport_name: a.id for a in session.query(Airport).all()}
    end_airport_id_map = start_airport_id_map.copy()

    # Creating flights
    airline_flights = []
    batch_size = 1000

    for _, row in df.iterrows():
        airline_id = airline_id_map.get(str(row['Airline']))
        flight_number_id = flight_number_id_map.get(str(row['Flight']))
        start_airport_id = start_airport_id_map.get(str(row['AirportFrom']))
        end_airport_id = end_airport_id_map.get(str(row['AirportTo']))

        if airline_id and flight_number_id and start_airport_id and end_airport_id:
            airline_flights.append(AirlineFlight(
                airline_id=airline_id,
                flight_number_id=flight_number_id,
                start_airport_id=start_airport_id,
                end_airport_id=end_airport_id,
                day_of_week=int(row['DayOfWeek']) if pd.notna(row['DayOfWeek']) else None,
                time=int(row['Time']) if pd.notna(row['Time']) else None,
                length=int(row['Length']) if pd.notna(row['Length']) else None,
                delay=bool(row['Delay']) if pd.notna(row['Delay']) else None
            ))

        if len(airline_flights) >= batch_size:
            session.add_all(airline_flights)
            session.commit()
            airline_flights = []

    if airline_flights:
        session.add_all(airline_flights)
        session.commit()


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('airlines', sa.Column('delay', sa.BOOLEAN(), autoincrement=False, nullable=True))
    op.add_column('airlines', sa.Column('length', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('airlines', sa.Column('delay_probability', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('airlines', sa.Column('airport_from', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('airlines', sa.Column('time', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('airlines', sa.Column('flight_number', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('airlines', sa.Column('day_of_week', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('airlines', sa.Column('airline', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('airlines', sa.Column('start_at', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('airlines', sa.Column('airport_to', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.drop_constraint(None, 'airlines', type_='unique')
    op.drop_column('airlines', 'name')
    op.drop_table('airline_flights')
    op.drop_table('flight_numbers')
    op.drop_table('airports')
    # ### end Alembic commands ###
