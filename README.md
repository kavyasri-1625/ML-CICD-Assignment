# Iris ML CI/CD Pipeline

## Project Overview
This project implements a CI/CD pipeline for a Machine Learning model using GitHub Actions and the Iris dataset.

## Technologies Used
- Python and Scikit-learn
- Pytest
- GitHub Actions
- GitHub Pages
- HTML and CSS

## Model Details
- **Algorithm:** Logistic Regression
- **Dataset:** Iris (150 samples, 4 features, 3 classes)
- **Evaluation:** Accuracy score and automated tests

## CI/CD Workflow
1. Install dependencies.
2. Train the ML model.
3. Run automated tests using pytest.
4. Deploy the webpage if CI passes.

## Project Structure
- `model.py` – Model training and prediction
- `test_model.py` – Automated tests
- `requirements.txt` – Required packages
- `index.html` – Webpage
- `.github/workflows/cicd.yml` – CI/CD workflow

## Expected Outcome
An automated pipeline that trains and tests the Iris model and deploys a webpage through GitHub Pages.
