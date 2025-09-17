# Sepsis Prediction API

A FastAPI-based backend service for predicting sepsis risk and generating medical reports from patient vitals data.

## Features

- **Sepsis Risk Prediction**: Predicts the likelihood of sepsis based on patient vitals
- **Batch Processing**: Process multiple patient records via CSV upload
- **Report Generation**: Generate detailed PDF reports with prediction results
- **SHAP Explanation**: Includes feature importance analysis using SHAP values
- **User Authentication**: Secure user registration and login
- **Email Notifications**: Send prediction reports via email

## Tech Stack

- **Backend**: FastAPI
- **Machine Learning**: TensorFlow, Keras
- **Data Processing**: Pandas, NumPy
- **Database**: SQLAlchemy ORM
- **Authentication**: JWT (JSON Web Tokens)
- **Documentation**: OpenAPI (automatically generated at `/docs`)

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sepsis_backend.git
   cd sepsis_backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory with the following variables:
   ```
   DATABASE_URL=sqlite:///./sepsis.db
   SECRET_KEY=your-secret-key
   ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   SMTP_SERVER=your-smtp-server
   SMTP_PORT=587
   SMTP_USERNAME=your-email@example.com
   SMTP_PASSWORD=your-email-password
   OPENAI_API_KEY=your_openai_api_key
   SENDGRID_API_KEY=your_sendgrid_api_key
   FROM_EMAIL=your_verified_sender@example.com
   ```

## Running the Application

1. Start the FastAPI development server:
   ```bash
   uvicorn app.main:app --reload
   ```

2. The API will be available at `http://localhost:8000`

3. Access the interactive API documentation:
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Authentication
- `POST /api/v1/register` - Register a new user
- `POST /api/v1/token` - Get access token (login)

### Predictions
- `POST /api/v1/csv_predict` - Upload CSV file for batch predictions
- `POST /api/v1/manual_predict` - Submit individual patient data for prediction
- `POST /api/v1/generate_batch_reports` - Generate reports for multiple patients
- `POST /api/v1/send_report` - Send report via email

## Project Structure

```
sepsis_backend/
├── app/
│   ├── api/               # API routes
│   ├── core/              # Core configurations
│   ├── models/            # Database models
│   ├── my_database/       # Database configuration and migrations
│   ├── schemas/           # Pydantic models for request/response
│   ├── services/          # Business logic and ML services
│   └── main.py            # Application entry point
├── env/                   # Virtual environment
├── plots/                 # Generated plots and reports
├── .env                  # Environment variables
└── requirements.txt      # Project dependencies
```

## Example Usage

### Making a Prediction

```python
import requests

# Example patient data
data = {
    "age": 65,
    "heart_rate": 90,
    "temperature": 38.5,
    "respiratory_rate": 22,
    "wbc_count": 12000,
    "platelets": 150000,
    "creatinine": 1.2,
    "lactate": 2.1
}

# Make prediction
response = requests.post(
    "http://localhost:8000/api/v1/manual_predict",
    json=data
)

print(response.json())
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

For any questions or feedback, please contact [Your Name] at [your.email@example.com]
