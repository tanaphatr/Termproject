# Datafile/load_data.py
import pandas as pd
from sqlalchemy import create_engine
from Utils.config import DATABASE_CONFIG  # Ensure you have the DATABASE_CONFIG correctly set

def load_data():
    # Create SQLAlchemy engine
    engine = create_engine(DATABASE_CONFIG)

    # SQL query to load data
    query = "SELECT * FROM salesdata"
    
    # Read data from database
    df = pd.read_sql(query, engine)
    return df
