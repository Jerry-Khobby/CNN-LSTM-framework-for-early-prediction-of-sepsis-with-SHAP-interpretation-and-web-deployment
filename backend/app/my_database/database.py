# app/database/database.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get the database URL
DATABASE_URL = os.getenv("DATABASE_URL")

# Step 1: Automatically create directory if it doesn't exist
if DATABASE_URL.startswith("sqlite:///"):
    # Extract the path after "sqlite:///"
    db_path = DATABASE_URL.replace("sqlite:///", "", 1)
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

# Step 2: Create engine and session
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Step 3: Declare Base
Base = declarative_base()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
