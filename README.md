# Sepsis Prediction in Hospitalized Patients

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![React](https://img.shields.io/badge/React-18.x-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0-009688)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Backend API](#-backend-api)
- [Frontend Interface](#-frontend-interface)
- [Notebooks](#-notebooks)
- [Development Workflow](#-development-workflow)
- [Testing & Evaluation](#-testing--evaluation)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

---

## 🌟 Project Overview

Sepsis is a severe, life-threatening condition that arises when the body's response to infection causes injury to its own tissues and organs. Early detection and intervention are crucial for improving patient outcomes and reducing mortality rates. This project aims to provide a robust, end-to-end solution for predicting sepsis onset in hospitalized patients using advanced deep learning techniques applied to clinical time-series data.

The solution integrates a hybrid CNN-LSTM model, a RESTful API backend, and a modern web dashboard. It is designed for real-world hospital environments, supporting both automated and manual data input, secure data management, and interpretable predictions. The modular architecture ensures scalability and ease of integration with existing healthcare systems.

---

## ✨ Key Features

- **Early Sepsis Detection:** Predicts the risk of sepsis hours before clinical diagnosis, enabling timely intervention and improved patient care.
- **Hybrid Deep Learning Model:** Utilizes Convolutional Neural Networks (CNN) for extracting spatial features and Long Short-Term Memory (LSTM) networks for capturing temporal dependencies in patient data.
- **Class Imbalance Handling:** Implements advanced techniques such as focal loss, class weighting, and Synthetic Minority Over-sampling Technique (SMOTE) to address the imbalance commonly found in medical datasets.
- **Web-based Dashboard:** Provides an intuitive React-powered interface for clinicians to upload data, view predictions, analyze model metrics, and export reports.
- **RESTful API:** FastAPI backend enables seamless integration with hospital information systems and third-party applications.
- **Interpretability:** Incorporates SHAP (SHapley Additive exPlanations) for feature importance visualization, enhancing clinical trust and transparency.
- **Secure Data Management:** Uses SQLite for storing patient records, with token-based authentication to ensure data privacy and security.
- **Extensible Design:** Modular codebase allows for easy customization, extension, and adaptation to different clinical settings or datasets.
- **Automated Reporting:** Generates downloadable PDF and CSV reports for documentation and further analysis.
- **Comprehensive Evaluation:** Includes ROC, PR curves, and confusion matrix plots for thorough model assessment.

---

## 🏗️ Project Structure

```
Cnn-lstm-sepsis-prediction/
├── backend/                  # FastAPI backend, model serving, database
│   ├── app/
│   │   ├── api/              # API endpoints (routes.py)
│   │   ├── core/             # Config, scaling, utilities
│   │   ├── models/           # Saved Keras model
│   │   ├── my_database/      # SQLite DB, ORM models, tokens
│   │   ├── schemas/          # Pydantic schemas for requests/responses
│   │   ├── services/         # Prediction, reporting, SHAP explanations
│   │   └── plots/            # Model evaluation plots (ROC, PR, confusion matrix)
│   └── requirements.txt      # Python dependencies
│
├── frontend/                 # React (Next.js) web interface
│   ├── app/                  # Pages: upload, results, export, about
│   ├── components/           # UI components: uploaders, cards, charts, PDF export
│   ├── hooks/, lib/, styles/ # Custom hooks, utilities, global styles
│   └── public/               # Static assets
│   └── package.json          # Frontend dependencies
│
├── model_training/           # Jupyter notebooks for model development
│   ├── folk-of-prediction-of-sepsis-time-series-main-notebook.ipynb
│   ├── handling-imbalance.ipynb
│   └── sepsis-prediction-from-clinical-data-trying-other-model-training-method.ipynb
```

---

## 🧠 Model Architecture

The core prediction engine is a hybrid CNN-LSTM neural network, specifically designed for sequential clinical data analysis:

1. **Input Layer:** Accepts multivariate time-series data, including vital signs and laboratory results for each patient over time.
2. **Convolutional Layers:** Multiple 1D CNN layers extract local temporal features, using ReLU activation, batch normalization, and dropout for regularization. These layers help in identifying short-term patterns and anomalies in the data.
3. **LSTM Layers:** Bidirectional LSTM layers capture long-term dependencies and trends in patient data, essential for understanding disease progression. An attention mechanism is integrated to focus on the most critical time steps.
4. **Dense Layers:** Fully connected layers aggregate extracted features for final prediction, with dropout and L2 regularization to prevent overfitting.
5. **Output Layer:** Sigmoid activation function provides a probability score for binary classification (sepsis risk).

**Model Innovations:**
- **Custom Loss Function:** Focal loss is used to address class imbalance, ensuring the model pays more attention to hard-to-classify cases.
- **Attention Mechanism:** Enhances interpretability by highlighting which time steps and features contributed most to the prediction.
- **Learning Rate Scheduling:** Adaptive optimizers and learning rate schedules improve training stability and performance.
- **Feature Importance:** SHAP explanations are generated to provide clinicians with insights into the most influential features for each prediction.

---

## 📊 Dataset

The model is trained and validated on a comprehensive clinical dataset, which includes:

- **Vital Signs:** Heart rate, oxygen saturation, temperature, blood pressure, respiratory rate, and more.
- **Laboratory Results:** White blood cell count (WBC), platelets, creatinine, bilirubin, lactate, and other key biomarkers.
- **Demographics:** Age, gender, and other relevant patient information.
- **Clinical Outcomes:** Binary sepsis label indicating the presence or absence of sepsis.

**Preprocessing Pipeline:**
- **Missing Values:** Forward-fill for time-series vitals, median imputation for laboratory results, and advanced imputation strategies for critical variables.
- **Feature Engineering:** Creation of time-based trends, statistical aggregations (mean, standard deviation, minimum, maximum), and clinical risk scores.
- **Normalization:** Standard scaling for continuous features and min-max scaling for select measurements to ensure uniformity.
- **Class Imbalance:** Addressed using SMOTE for oversampling, class weighting during training, and focal loss in the model.

**Data Security:** All patient data is handled in compliance with privacy standards, and the database is secured with token-based authentication.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Node.js 16 or higher
- pip (Python package manager)
- virtualenv (recommended for Python environments)

### Backend Setup
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install backend dependencies
cd backend
pip install -r requirements.txt
```

### Frontend Setup
```bash
# Install Node.js dependencies
cd frontend
npm install
```

---

## 🚀 Usage

### 1. Model Training
- Navigate to `model_training/`
- Open and run `folk-of-prediction-of-sepsis-time-series-main-notebook.ipynb` in Jupyter Notebook.
- The trained model will be saved as `model.keras` in `backend/app/models/`.

### 2. Running the Backend API
```bash
cd backend
uvicorn app.main:app --reload
```
The API will be available at `http://localhost:8000`. You can interact with the API using Swagger UI at `/docs`.

### 3. Running the Frontend
```bash
cd frontend
npm start
```
The web interface will be accessible at `http://localhost:3000`.

---

## 📡 Backend API

The FastAPI backend provides a range of endpoints for prediction, data management, and reporting.

### Predict Sepsis Risk
```
POST /api/predict
```
**Request Body:**
```json
{
  "patient_id": "12345",
  "vital_signs": [{"hr": 85, "o2sat": 98, "temp": 37.2, ...}],
  "lab_results": [{"wbc": 8.5, "platelets": 250, ...}],
  "demographics": {"age": 65, "gender": "M"}
}
```
**Response:**
```json
{
  "patient_id": "12345",
  "sepsis_risk": 0.76,
  "prediction": "High Risk",
  "confidence": 0.87,
  "important_features": [
    {"feature": "lactate", "importance": 0.92},
    {"feature": "resp_rate", "importance": 0.88}
  ]
}
```

### Other Endpoints
- **Manual Data Entry:** Submit patient data via form.
- **CSV Upload:** Bulk upload patient records for batch prediction.
- **Export Results:** Download prediction results as PDF or CSV.
- **Model Metrics:** Retrieve evaluation metrics and plots.

**Authentication:** Secure endpoints require token-based authentication for data privacy.

---

## 🖥️ Frontend Interface

The React (Next.js) frontend offers a user-friendly dashboard for clinicians and researchers:

- **Upload CSV/Manual Entry:** Easily input patient data via CSV files or manual forms.
- **Prediction Results:** Instantly view risk scores, confidence levels, and feature importance for each patient.
- **Export Reports:** Download results in PDF or CSV format for documentation and sharing.
- **Model Metrics:** Interactive visualizations of ROC curves, precision-recall curves, and confusion matrices.
- **About & Documentation:** Access detailed information about the model, its development, and usage instructions.
- **Responsive Design:** Optimized for desktop and mobile devices.

---

## 📓 Notebooks

- **Main Notebook:** Comprehensive pipeline for data preprocessing, model training, evaluation, and visualization.
- **Handling Imbalance:** Explores techniques such as SMOTE and focal loss to address class imbalance.
- **Alternative Architectures:** Experiments with different model architectures, hyperparameter tuning, and ablation studies.

These notebooks serve as both documentation and reproducible research for model development.

---

## 🧑‍💻 Development Workflow

1. **Clone the repository:**  
   `git clone https://github.com/yourusername/Cnn-lstm-sepsis-prediction.git`
2. **Set up backend and frontend as described above.**
3. **Develop features in branches:**  
   `git checkout -b feature/your-feature`
4. **Commit and push changes:**  
   `git commit -m "Describe your change"`  
   `git push origin feature/your-feature`
5. **Open a Pull Request for review.**

**Coding Standards:**  
- Follow PEP8 for Python code.
- Use ESLint and Prettier for JavaScript/TypeScript code.
- Write clear, concise commit messages.

**Documentation:**  
- Update README and code comments for new features.
- Add docstrings to Python functions and classes.

---

## 🧪 Testing & Evaluation

- **Backend:** Includes unit tests for API endpoints, model inference, and database operations. Use `pytest` for running tests.
- **Frontend:** Component and integration tests using Jest and React Testing Library.
- **Model:** Evaluation metrics such as Area Under the Curve (AUC), ROC curves, precision-recall curves, and confusion matrix plots are generated and stored in `backend/app/plots/`.

**Continuous Integration:**  
Set up CI pipelines for automated testing and deployment.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to your branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Contribution Guidelines:**
- Ensure code is well-documented and tested.
- Follow the project's coding standards.
- Describe your changes clearly in the Pull Request.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Research team for clinical dataset collection and annotation.
- Open-source contributors to TensorFlow, PyTorch, FastAPI, React, and supporting libraries.
- Medical professionals for providing domain expertise and feedback.
- The open-source community for tools, resources, and inspiration.

---

## 📧 Contact

For questions, suggestions, or feedback, please contact:

**Jeremiah Anku Coblah**  
Email: [jeremiah.anku.coblah@gmail.com]

---

## 📚 Further Reading & Resources

- [Sepsis Guidelines - Surviving Sepsis Campaign](https://www.sccm.org/SurvivingSepsisCampaign/Home)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
- [SMOTE for Imbalanced Data](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

---

## 🏥 Real-World Applications

This project can be adapted for:
- Hospital EHR integration for real-time sepsis alerts.
- Research studies on sepsis prediction and prevention.
- Educational tools for medical students and data scientists.
- Benchmarking and comparison with other clinical risk models.

---

## 🛡️ Data Privacy & Ethics

All patient data used in this project is handled in accordance with ethical guidelines and privacy regulations. No personally identifiable information is stored or shared. For real-world deployment, ensure compliance with HIPAA, GDPR, or relevant local regulations.

---

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for a history of notable changes and updates.
