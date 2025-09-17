from sqlalchemy import Column, Integer, String, Enum as SQLAEnum 
from app.my_database.database import Base
import enum

class UserRole(enum.Enum):
    doctor = "Doctor"
    nurse = "Nurse"
    physician_assistant = "Physician Assistant"
    nurse_practitioner = "Nurse Practitioner"
    resident = "Resident"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    role = Column(SQLAEnum(UserRole), nullable=False)
    hospital = Column(String, nullable=False)
    medical_license_number = Column(String, nullable=True)
    password = Column(String, nullable=False)
