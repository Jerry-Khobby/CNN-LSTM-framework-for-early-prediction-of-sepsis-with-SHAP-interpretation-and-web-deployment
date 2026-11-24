# Early Prediction of Sepsis with a CNN-LSTM Framework

A research project on hourly sepsis prediction for intensive care patients, comprising a leakage-controlled training pipeline, a CNN-LSTM ensemble, a LightGBM baseline, model interpretation, and a prototype web application (FastAPI and Next.js).

> **Intended use.** This is research and educational software. It has not been prospectively or externally validated and is not a medical device. It must not be used to guide patient care.

## Contents

1. [Summary](#summary)
2. [Project status](#project-status)
3. [Repository structure](#repository-structure)
4. [Dataset and prediction task](#dataset-and-prediction-task)
5. [Methodology](#methodology)
6. [Results](#results)
7. [Assessment](#assessment)
8. [Limitations](#limitations)
9. [Reproducing the results](#reproducing-the-results)
10. [Web application](#web-application)
11. [Known issues](#known-issues)
12. [Data availability, ethics, and license](#data-availability-ethics-and-license)
13. [References](#references)
14. [Contact](#contact)

## Summary

At every ICU hour, the model estimates the probability that the PhysioNet 2019 sepsis label is positive, using only data recorded up to that hour. Results on 6,051 held-out patients (440 septic; 233,183 patient-hours), with 95% patient-level bootstrap confidence intervals:

| Model | AUROC | AUPRC | Normalized utility |
|---|---|---|---|
| CNN-LSTM ensemble (tuned) | 0.830 (0.812-0.847) | 0.118 (0.102-0.142) | 0.398 (0.355-0.440) |
| LightGBM baseline | 0.840 (0.824-0.858) | 0.117 (0.102-0.137) | 0.416 (0.377-0.454) |
| Blend of both | 0.841 (0.824-0.857) | 0.125 (0.108-0.150) | 0.418 (0.376-0.459) |

An uninformative classifier has an AUPRC of about 0.018, the proportion of positive hours. Utility is reported at the threshold that maximized it on validation data.

The CNN-LSTM discriminates well above chance but does not outperform LightGBM: its AUROC is lower, and no difference in AUPRC or utility was detected. Blending gives a small gain in AUPRC only.

## Project status

| Component | Status |
|---|---|
| Training pipeline | Complete; two full runs on Kaggle |
| Reported models | Produced by the notebooks; model files are not stored in this repository |
| Web application | Prototype; **serves an earlier model, not the models reported above** |
| External validation | Not performed |

The deployed file `backend/app/models/model.keras` (saved May 2025) is a bidirectional LSTM with no convolutional layers. It takes 10 hours of five vital signs and 33 summary values. None of the results in this document describe that model.

## Repository structure

```text
.
|-- backend/                     FastAPI service
|   |-- app/
|   |   |-- api/routes.py        API endpoints
|   |   |-- core/                Configuration and scaling parameters
|   |   |-- models/model.keras   Earlier deployed model (BiLSTM)
|   |   |-- my_database/         SQLAlchemy models and token helper
|   |   |-- schemas/             Request schemas, scaling, evaluation
|   |   `-- services/            Prediction, SHAP, report generation
|   |-- plots/                   Plots from the earlier model
|   `-- requirements.txt
|-- frontend/                    Next.js application
`-- model_training/              Jupyter notebooks
```

| Notebook in `model_training/` | Purpose |
|---|---|
| `sepsis-cnn-lstm-tuned-pipeline.ipynb` | **Current pipeline (Run 2).** Source only; the executed copy is not yet in the repository |
| `sepsis-cnn-lstm-improved-pipeline.ipynb` | Run 1 source |
| `notebook3d6d430314.ipynb` | Run 1 as executed on Kaggle, with outputs |
| `folk-of-prediction-of-sepsis-time-series-main-notebook.ipynb` | Original LSTM and CNN-LSTM experiments (superseded) |
| `handling-imbalance.ipynb` | Original resampling and focal-loss experiments (superseded) |
| `sepsis-prediction-from-clinical-data-trying-other-model-training-method.ipynb` | Original SMOTE and focal-loss experiments (superseded) |

The superseded notebooks use different cohorts, labels, and splits, and contain methodological errors, including window-level validation splits and future-value imputation. Their metrics are not directly comparable with those reported here.

## Dataset and prediction task

The data come from the [PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/).

| Property | Value |
|---|---|
| Patients | 40,336 (hospital A: 20,336; hospital B: 20,000) |
| Hourly records | 1,552,210 |
| Septic patients | 2,932 (7.27%) |
| Positive hours | 27,916 (1.80%) |
| Sepsis rate by hospital | A: 8.80%; B: 5.71% |
| Median stay | 38 hours |

Variables comprise 8 vital signs, 26 laboratory values, and demographic and admission fields. Most laboratory values are missing at any given hour.

In septic patients, `SepsisLabel` becomes positive six hours before the challenge-defined onset time. Predicting the label at hour *t* from data up to hour *t* is therefore an early-prediction task. An exception is the 14.5% of septic patients who are already positive in their first recorded hour, for whom early warning within the stay is limited.

## Methodology

### Patient-level split

Patients were assigned to a single partition, stratified by sepsis status and hospital.

| Partition | Patients | Septic | Hours | Use |
|---|---|---|---|---|
| Training | 28,234 | 2,052 | 1,087,335 | Fitting models and preprocessing |
| Validation | 6,051 | 440 | 231,692 | Early stopping, model selection, blending, thresholds |
| Test | 6,051 | 440 | 233,183 | Evaluation |

### Features

Each hour is represented by 173 features computed from current and past data of the same patient:

- Last observed value of each physiological variable, carried forward
- Number of measurements so far and hours since the last measurement
- Six-hour mean, minimum, maximum, and standard deviation, with one-hour and six-hour changes, for seven vital signs
- Deviation from the patient's 24-hour and whole-stay baseline (to date) for six vital signs
- Shock index, pulse pressure, BUN/creatinine ratio, SIRS count, partial qSOFA and SOFA, and lactate and MAP flags
- Age, sex, ICU unit, hospital admission time, and ICU length of stay to date

An automated test truncates the records of 40 patients, recomputes the features, and confirms that none change. Values are clipped to training percentiles (0.5 and 99.5) and standardized with training statistics. Remaining missing values are set to the training mean.

### CNN-LSTM

Input is a 24-hour window of 174 channels (173 features and a padding indicator). Windows are zero-padded at the start of a stay so that every hour is scored.

```text
Input (24 x 174)
  Spatial dropout and Gaussian noise
  Causal Conv1D (64), batch normalization, ReLU
  Causal dilated Conv1D (64, dilation 2), batch normalization, ReLU
  Spatial dropout
  LSTM (64), then LSTM (32)
  Last hidden state concatenated with attention-pooled states
  Dense (32), dropout, sigmoid output
```

Training uses binary cross-entropy with AdamW and gradient clipping. Each pass contains all positive hours and a new 30% sample of negative hours; predicted log-odds are then corrected for the sampling rate. Early stopping monitors validation AUPRC.

Ten configurations were compared on validation AUPRC, varying learning rate, weight decay, network size, regularization, negative sampling, and window length. The selected configuration:

| Parameter | Value |
|---|---|
| Window length | 24 hours |
| Network size | Small (as above) |
| Learning rate; weight decay | 3e-4; 1e-3 |
| Input dropout; hidden dropout | 0.2; 0.5 |
| Gaussian noise (standard deviation) | 0.1 |
| Negative sampling fraction | 0.30 |

The notebook retrains the selected configuration with five seeds and averages the predictions.

### Baseline and blend

LightGBM is trained on the same 173 features for the current hour, with early stopping on validation AUPRC but no hyperparameter search. The blend is a weighted average of the two models' probabilities, with the weight chosen on validation data.

### Evaluation

Three thresholds are fixed on validation data: maximum F1, maximum normalized challenge utility, and at least 80% hourly sensitivity. Confidence intervals and paired differences use 200 patient-level bootstrap resamples. The utility implementation was checked numerically against a direct loop implementation of the challenge scoring rule.

## Results

### Comparison with the baseline

Paired differences on the same resampled patients. Differences whose intervals exclude zero are in bold. P-values have a resolution of 0.01 and are not adjusted for multiple comparisons.

| Comparison | Metric | Difference | 95% CI | p |
|---|---|---|---|---|
| CNN-LSTM minus LightGBM | AUROC | **-0.0101** | -0.0195 to -0.0015 | < 0.01 |
| CNN-LSTM minus LightGBM | AUPRC | 0.0010 | -0.0095 to 0.0136 | 0.80 |
| CNN-LSTM minus LightGBM | Utility | -0.0177 | -0.0398 to 0.0055 | 0.12 |
| Blend minus LightGBM | AUROC | 0.0007 | -0.0047 to 0.0057 | 0.77 |
| Blend minus LightGBM | AUPRC | **0.0078** | 0.0017 to 0.0164 | < 0.01 |
| Blend minus LightGBM | Utility | 0.0022 | -0.0115 to 0.0189 | 0.77 |

### Operating points (hourly)

| Model | Threshold | Sensitivity | Specificity | PPV | F1 | Utility |
|---|---|---|---|---|---|---|
| CNN-LSTM | Maximum F1 | 0.299 | 0.965 | 0.135 | 0.187 | 0.253 |
| CNN-LSTM | Maximum utility | 0.639 | 0.842 | 0.069 | 0.125 | 0.398 |
| CNN-LSTM | 80% sensitivity | 0.814 | 0.683 | 0.045 | 0.085 | 0.310 |
| LightGBM | Maximum F1 | 0.275 | 0.970 | 0.143 | 0.188 | 0.236 |
| LightGBM | Maximum utility | 0.666 | 0.836 | 0.069 | 0.125 | 0.416 |
| LightGBM | 80% sensitivity | 0.803 | 0.716 | 0.049 | 0.093 | 0.358 |
| Blend | Maximum F1 | 0.318 | 0.962 | 0.135 | 0.189 | 0.268 |
| Blend | Maximum utility | 0.650 | 0.847 | 0.072 | 0.130 | 0.418 |
| Blend | 80% sensitivity | 0.807 | 0.711 | 0.049 | 0.092 | 0.352 |

### Patient-level alarms (CNN-LSTM)

A septic patient counts as detected if any alarm occurs from 48 hours before to 3 hours after the derived onset time (six hours after the first positive label).

| Threshold | Septic patients detected | Median lead time (hours) | Non-septic patients with any alarm | Alarm-hours per non-septic patient-day |
|---|---|---|---|---|
| Maximum F1 | 196 of 440 (44.5%) | 24 | 4.1% | 0.18 |
| Maximum utility | 373 of 440 (84.8%) | 22 | 34.6% | 2.62 |
| 80% sensitivity | 434 of 440 (98.6%) | 27 | 89.4% | 6.45 |

Lead times are computed among detected patients and measure when an alarm first occurred, not demonstrated clinical benefit. At the 80% sensitivity threshold the model alarms at some point for almost every patient, septic or not, so that setting does not separate the two groups at patient level.

### Effect of tuning

| CNN-LSTM | AUROC | AUPRC | Utility |
|---|---|---|---|
| Run 1 (untuned, 3 seeds) | 0.826 | 0.116 | 0.384 |
| Run 2 (tuned) | 0.830 | 0.118 | 0.398 |
| Change | +0.004 | +0.002 | +0.014 |

Both runs use the same test patients. All three Run 1 seeds reached their best validation score after one epoch and then overfitted, which motivated Run 2. The differences between runs were not tested for significance and are small relative to the confidence intervals.

### Variable importance

Drop in validation AUPRC when all features derived from a variable are permuted, for the ten highest-ranked variables:

| Variable | Drop | Variable | Drop |
|---|---|---|---|
| Temperature | 0.0264 | Heart rate | 0.0051 |
| ICU length of stay | 0.0199 | Respiratory rate | 0.0049 |
| FiO2 | 0.0093 | pH | 0.0043 |
| Oxygen saturation | 0.0084 | Mean arterial pressure | 0.0036 |
| Systolic pressure | 0.0067 | Log ICU length of stay | 0.0033 |

These values come from a validation subsample enriched in positive hours and should be used for ranking only. Correlated variables share importance. SHAP values are computed for the LightGBM baseline, not for the CNN-LSTM.

## Assessment

- **Discrimination is moderate to good.** An AUROC of 0.83 and an AUPRC about 6.5 times the base rate show a real signal.
- **The CNN-LSTM is not the best model.** It is below a single, untuned LightGBM model on AUROC, despite a ten-configuration search and ensembling. The results do not support a claim that the deep architecture is superior.
- **Tuning helped little.** Similar results from two different model families suggest, but do not prove, that the available features and labels limit performance.
- **Alarm precision is low.** At the maximum-utility threshold about 1 in 14 alarm-hours is a true positive, and 34.6% of non-septic patients receive at least one alarm. At the maximum-F1 threshold precision is about 1 in 7, but more than half of septic patients are missed.
- **The evidence supports** a methodological study of leakage-controlled sepsis prediction with a transparent baseline comparison. It does not support clinical deployment.

## Limitations

- **Repeated use of the test set.** The same test patients were evaluated in both runs, and Run 1 test results were known before Run 2 was designed. Run 2 settings were selected on validation data only, but its test results are not a fully independent confirmation. Statements in the notebooks that the test set is used only once apply to a single run.
- **Validation reuse.** Early stopping, configuration selection, blending, and thresholds share one validation set.
- **No external validation.** Both hospitals appear in every partition. Transfer to a new hospital was not tested, and no per-hospital results are reported.
- **Length-of-stay dependence.** ICU length of stay is the second most important variable and may partly reflect cohort construction.
- **Onset and lead time.** For patients positive in their first hour, the true onset may be earlier than the derived onset. Septic records end at most nine hours after the first positive label, so the 48-hour detection window often covers the whole stay.
- **Unequal comparison.** The CNN-LSTM was tuned and ensembled; LightGBM was not tuned.
- **Uncertainty estimates.** Intervals reflect test-patient sampling only, not retraining variability, and use 200 resamples.
- **Missing predictors.** Antibiotics, cultures, fluids, consciousness scores, and clinical notes are unavailable.

## Reproducing the results

1. Obtain the Challenge 2019 data in the single-CSV layout used here (hourly rows with `Patient_ID`, `ICULOS`, and `SepsisLabel`). The dataset is not stored in this repository.
2. Upload `model_training/sepsis-cnn-lstm-tuned-pipeline.ipynb` to Kaggle and attach the dataset.
3. Set `KAGGLE_DATA_PATH` in Section 2 if needed. Otherwise the notebook searches `/kaggle/input`.
4. Select a T4 GPU and run all cells.
5. Confirm that the output reports a passed leakage check, split check, and reload check.
6. Download `sepsis_artifacts.zip`.

The archive contains the trained models, preprocessing parameters, configuration and thresholds, patient split, test records and predictions, result tables, figures, and `sepsis_pipeline.py`, whose `predict_raw()` function scores raw hourly records. GPU training is not fully deterministic, so small numerical differences are expected.

## Web application

| Layer | Technology |
|---|---|
| Frontend | Next.js 15, React 19, TypeScript, Tailwind CSS |
| Backend | FastAPI, TensorFlow/Keras, SHAP, SQLAlchemy |
| Batch report text | xAI API (external service) |

### API endpoints

All routes are prefixed with `/api/v1`. Interactive documentation is available at `/docs` while the server is running.

| Method and path | Purpose |
|---|---|
| `POST /csv_predict` | Score a preprocessed CSV containing `X_cont` and `label` columns |
| `POST /manual_predict` | Score one patient from 10 hourly values of five vital signs and 33 other fields |
| `POST /reports/batch` | Generate narrative reports through the xAI API |
| `POST /send-report` | Email a PDF report |
| `POST /register` | Register a user (see [Known issues](#known-issues)) |

### Local setup

Backend (Windows):

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install sqlalchemy passlib bcrypt "python-jose[cryptography]" xai-sdk
uvicorn app.main:app --reload
```

The second `pip install` adds packages that the code uses but `requirements.txt` omits. Create `backend/.env` before starting the server, and do not commit it:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Required, for example `sqlite:///app/my_database/hospital.db` |
| `SECRET_KEY`, `ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES` | Token creation; the last two have defaults |
| `GROK_API_KEY` | xAI API key for batch reports |
| `EMAIL_USER`, `EMAIL_PASSWORD`, `EMAIL_HOST`, `EMAIL_PORT` | Report email; host and port have defaults |

Frontend:

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at `http://localhost:3000` and expects the API at `http://127.0.0.1:8000` (`frontend/lib/constant.ts`). These steps were derived from the source code and have not been verified on a clean machine.

## Known issues

- **Unverified values in the interface.** The About page states a test AUC of 0.7615 and agreement between validation and test performance, and the dataset panel (`frontend/components/dataset-info.tsx`) states demographics, care settings, and a data split. None of these could be traced to the notebooks in this repository.
- **Outdated model.** The backend serves the earlier BiLSTM, while the About page describes a CNN-LSTM.
- **No access control.** Prediction, report, and email endpoints require no authentication, and CORS allows all origins.
- **Registration fails.** `/register` uses `dcrypt_context`, which is not defined, and the included database contains no tables.
- **External data transfer.** `/reports/batch` is designed to send prediction details to a third-party API. Do not submit identifiable data.
- **Labels required for CSV scoring.** `/csv_predict` requires outcome labels and derives its threshold from the uploaded file. It returns individual predictions for the first 20 rows only.
- **Explanations in manual mode.** The reported key drivers are the largest standardized inputs, not SHAP values, and the generated narrative presents standardized values with clinical units.
- **Treatment text.** Manual mode returns fixed medication and dosing text linked to fixed risk cut-offs that were not derived from validation data. It should be disabled unless reviewed by qualified clinicians.
- **Platform.** Model paths in `backend/app/core/config.py` use Windows separators.
- **Tests.** The repository contains no automated application tests.

## Data availability, ethics, and license

The challenge data are distributed by PhysioNet under their own terms, which users must review. The project has not been assessed for compliance with HIPAA, GDPR, or similar regulations.

No license file is currently included, so no reuse rights are granted by default. Contact the author for permission.

## References

1. Reyna MA, et al. Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019. *Critical Care Medicine*. 2020;48:210-217. [PubMed](https://pubmed.ncbi.nlm.nih.gov/31939789/)
2. [PhysioNet Challenge 2019 data, version 1.0.0](https://physionet.org/content/challenge-2019/1.0.0/)
3. Ke G, et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*, 2017.
4. Lundberg SM, Lee SI. A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*, 2017.

## Contact

Jeremiah Anku Coblah, jeremiah.anku.coblah@gmail.com
