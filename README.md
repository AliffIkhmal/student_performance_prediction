# Student Performance Prediction System

A web app that predicts student grades (A–F) using machine learning — built to help lecturers and academic advisors spot at-risk students early and take action before it's too late.

This started as a Final Year Project and grew into a full Django portal with dashboards, intervention tracking, and an API layer.

---

## What It Does

The system takes in student data (things like study hours, absences, parental support, extracurriculars) and runs it through 8 different ML models to find the one that predicts grades most accurately. Once trained, you get a full dashboard to explore predictions, track interventions, and upload new datasets — all through a clean, role-based portal.

### Features

- **Smart model selection** — compares Random Forest, Gradient Boosting, SVM, KNN, Logistic Regression, Decision Tree, Extra Trees, and Neural Network, then picks the best one by F1 score
- **SMOTE for imbalanced data** — handles the fact that some grades (like A) are way underrepresented in real student data
- **SHAP explainability** — every prediction shows which factors mattered most and why, so it's not just a black box
- **Dashboard** — KPI cards, grade distribution, risk breakdown, scatter charts, and top students at a glance
- **Batch predictions** — upload a CSV of students and get predictions for all of them at once
- **Intervention tracking** — log what actions were taken for at-risk students, track outcomes, and export the history to CSV
- **Dataset management** — upload your own CSV with automatic column mapping, so you're not locked into one format
- **Role-based access** — admins manage users and settings; lecturers work with data and predictions
- **API with Bearer tokens** — secure endpoints if you want to hook predictions into other tools
- **Dark mode** — full light/dark toggle with system preference detection
- **Mobile-friendly** — responsive sidebar and bottom nav for smaller screens

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Backend | Django 5.1, Python |
| ML | scikit-learn, SMOTE (imbalanced-learn), SHAP, joblib |
| Frontend | Tailwind CSS, Chart.js, Django templates |
| Data | pandas, NumPy |
| Database | SQLite (dev), PostgreSQL (prod-ready via dj-database-url) |
| Deployment | Gunicorn, WhiteNoise |

---

## Getting Started

### Prerequisites

- Python 3.12+
- pip

### Setup

```bash
# Clone the repo
git clone https://github.com/AliffIkhmal/student_performance_prediction.git
cd student_performance_prediction

# Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Create an admin account
python manage.py createsuperuser

# Start the server
python manage.py runserver
```

Then open `http://127.0.0.1:8000` and sign in.

---

## How the ML Pipeline Works

1. **Split** — 80/20 train/test split
2. **Oversample** — SMOTE balances the training set so rare grades aren't ignored (applied only to training data, no data leakage)
3. **Scale** — StandardScaler normalizes features
4. **Compare** — all 8 models are trained and evaluated
5. **Select** — the model with the highest weighted F1 score wins
6. **Save** — the trained model is persisted with joblib for instant loading next time

---

## Project Structure

```
├── model.py                 # ML model class — training, comparison, prediction
├── app.py                   # Legacy Streamlit frontend (still works)
├── manage.py                # Django management
├── student_portal/          # Django project settings (base/dev/prod)
├── portal/                  # Main Django app
│   ├── views.py             # All views and API endpoints
│   ├── models.py            # UserProfile, Dataset, Interventions, API tokens
│   ├── services.py          # Business logic layer
│   ├── column_mapping.py    # Auto-maps uploaded CSV columns
│   └── tests.py             # Portal tests
├── templates/               # Django HTML templates
│   ├── base.html            # Base layout with dark mode + Tailwind config
│   ├── partials/            # Nav, sidebar, reusable components
│   └── portal/              # Page templates
├── static/portal/css/       # Custom CSS with CSS variables for theming
└── Student_performance_data_.csv  # Sample dataset (~2,392 rows)
```

---

## Production Deployment

Copy `.env.example` to `.env` and fill in your values:

```
DJANGO_ENV=production
DJANGO_SECRET_KEY=<generate-a-long-random-key>
DJANGO_ALLOWED_HOSTS=yourdomain.com
DJANGO_CSRF_TRUSTED_ORIGINS=https://yourdomain.com
```

Then collect static files and run with Gunicorn:

```bash
python manage.py collectstatic --noinput
python manage.py migrate
gunicorn student_portal.wsgi
```

---

## Running Tests

```bash
python manage.py test portal
```

---

## License

This project was built as a final year project. Feel free to explore the code.

---

## Project Structure

```
manage.py                    # Django entry point
student_portal/
  settings/
    base.py                  # Shared settings
    development.py           # DEBUG=True, relaxed security
    production.py            # Env-based secrets, HSTS, WhiteNoise, logging
portal/
  views.py                   # Route handlers
  services.py                # Business logic and ML integration
  models.py                  # UserProfile, ApiAccessToken, LecturerDataset, InterventionRecord
  forms.py                   # All form validation
  auth_utils.py              # Role helpers, API token management
  column_mapping.py          # Flexible CSV header alias resolution
  tests.py                   # 33 automated tests
model.py                     # StudentPerformanceModel (train, predict, save/load)
templates/                   # Django templates (base, partials, portal pages)
static/portal/css/           # site.css (theme tokens) + tailwind.min.css (built)
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `DJANGO_ENV` | Yes (prod) | Set to `production` to activate production settings |
| `DJANGO_SECRET_KEY` | Yes (prod) | Long random secret key |
| `DJANGO_ALLOWED_HOSTS` | Yes (prod) | Comma-separated hostnames |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | Yes (prod) | Comma-separated origins with scheme |
| `DJANGO_SECURE_SSL_REDIRECT` | No | `True` (default) or `False` |
| `DATABASE_URL` | No | PostgreSQL connection string; defaults to SQLite |

See `.env.example` for the full list.

---

## License

This project was built as an academic Final Year Project.
