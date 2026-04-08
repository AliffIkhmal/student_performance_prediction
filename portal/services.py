from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import pandas as pd
from django.contrib.auth import get_user_model
from django.db.models import Count
from django.utils import timezone

from model import StudentPerformanceModel
from .column_mapping import (
    BATCH_OPTIONAL_COLUMNS,
    BATCH_REQUIRED_COLUMNS,
    DATASET_REQUIRED_COLUMNS,
    apply_column_mapping,
    resolve_columns,
)
from .auth_utils import bootstrap_default_admin, get_user_role
from .models import InterventionRecord, LecturerDataset


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "Student_performance_data _.csv"
MODEL_PATH = BASE_DIR / "trained_model.pkl"

SUPPORT_LABELS = ["None", "Low", "Moderate", "High", "Very High"]
EDUCATION_LABELS = ["None", "High School", "Some College", "Bachelor's", "Higher"]
GENDER_LABELS = {0: "Male", 1: "Female"}
BOOLEAN_LABELS = {0: "No", 1: "Yes"}
FEATURE_LABELS = {
    "Age": "Age",
    "Gender": "Gender",
    "ParentalEducation": "Parental Education",
    "StudyTimeWeekly": "Study Time Weekly",
    "Absences": "Absences",
    "ParentalSupport": "Parental Support",
    "Extracurricular": "Extracurricular",
    "Sports": "Sports",
    "Music": "Music",
    "Volunteering": "Volunteering",
}
ACTIVITY_FEATURES = {
    "Extracurricular": "Extracurricular",
    "Sports": "Sports",
    "Music": "Music",
    "Volunteering": "Volunteering",
}
RECOMMENDATION_SEVERITY_STYLE_MAP = {
    InterventionRecord.Severity.URGENT: "bg-red-100 text-red-700",
    InterventionRecord.Severity.RECOMMENDED: "bg-amber-100 text-amber-700",
    InterventionRecord.Severity.OPTIONAL: "bg-sky-100 text-sky-700",
}

RISK_STYLE_MAP = {
    "High Risk": "bg-red-50 text-red-700 ring-1 ring-red-200",
    "Moderate": "bg-amber-50 text-amber-700 ring-1 ring-amber-200",
    "Low Risk": "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200",
}

ROLE_STYLE_MAP = {
    "admin": "bg-slate-900 text-white",
    "lecturer": "bg-sky-50 text-sky-700 ring-1 ring-sky-200",
}

PERIOD_DIMENSION_CONFIG = {
    "academicsession": {"label": "Academic Session", "type": "categorical"},
    "session": {"label": "Academic Session", "type": "categorical"},
    "semester": {"label": "Semester", "type": "categorical"},
    "term": {"label": "Term", "type": "categorical"},
    "intake": {"label": "Intake", "type": "categorical"},
    "year": {"label": "Academic Year", "type": "categorical"},
    "academicyear": {"label": "Academic Year", "type": "categorical"},
    "recorddate": {"label": "Record Date", "type": "date"},
    "date": {"label": "Record Date", "type": "date"},
    "createdat": {"label": "Created Date", "type": "date"},
    "updatedat": {"label": "Updated Date", "type": "date"},
}


class StudentNotFoundError(LookupError):
    pass


DEFAULT_FORM_VALUES = {
    "Age": 17,
    "Gender": 0,
    "ParentalEducation": 2,
    "StudyTimeWeekly": 10.0,
    "Absences": 5,
    "ParentalSupport": 2,
    "Extracurricular": 0,
    "Sports": 0,
    "Music": 0,
    "Volunteering": 0,
}

NUMERIC_DATASET_COLUMNS = [
    "StudentID",
    *StudentPerformanceModel.FEATURES,
    "GPA",
    "GradeClass",
]


def empty_period_filters(helper_text, active_label="All records"):
    return {
        "available": False,
        "dimensions": [],
        "selected_dimension": "",
        "selected_type": "",
        "value_options": [],
        "selected_value": "",
        "compare_options": [],
        "selected_compare_value": "",
        "date_from": "",
        "date_to": "",
        "min_date": "",
        "max_date": "",
        "compare_previous": False,
        "active_label": active_label,
        "helper_text": helper_text,
    }


def empty_dashboard_context(period_filters, title, message, dataset_info=None):
    return {
        "summary_cards": [],
        "grade_chart": {"labels": [], "counts": []},
        "risk_chart": {"labels": [], "counts": []},
        "study_scatter": [],
        "absences_scatter": [],
        "top_students": [],
        "feature_insights": [],
        "interventions": [],
        "model_snapshot": None,
        "period_filters": period_filters,
        "period_comparison": None,
        "intervention_analytics": None,
        "recent_interventions": [],
        "dataset_info": dataset_info,
        "dashboard_empty_title": title,
        "dashboard_empty_message": message,
    }


def build_dataset_info(dataset_record):
    if dataset_record is None:
        return None

    return {
        "id": dataset_record.pk,
        "original_filename": dataset_record.original_filename,
        "stored_filename": dataset_record.stored_filename,
        "row_count": dataset_record.row_count,
        "uploaded_at": dataset_record.uploaded_at,
        "confirmed_at": dataset_record.confirmed_at,
        "is_active": dataset_record.is_active,
    }


def get_intervention_queryset(user=None, include_all_for_admin=False):
    queryset = InterventionRecord.objects.select_related("user", "dataset")

    if user is None or not getattr(user, "is_authenticated", False):
        return queryset.none()

    if include_all_for_admin and get_user_role(user) == "admin":
        return queryset

    return queryset.filter(user=user)


def build_intervention_analytics(queryset, scope_label):
    today = timezone.localdate()
    total_count = queryset.count()
    active_count = queryset.filter(
        status__in=[InterventionRecord.Status.PLANNED, InterventionRecord.Status.IN_PROGRESS]
    ).count()
    completed_count = queryset.filter(status=InterventionRecord.Status.COMPLETED).count()
    reviewed_count = queryset.exclude(outcome=InterventionRecord.Outcome.PENDING).count()
    improved_count = queryset.filter(outcome=InterventionRecord.Outcome.IMPROVED).count()
    overdue_count = queryset.exclude(
        status__in=[InterventionRecord.Status.COMPLETED, InterventionRecord.Status.DISMISSED]
    ).filter(review_date__lt=today).count()
    improved_rate = int(round((improved_count / reviewed_count) * 100)) if reviewed_count else 0
    completion_rate = int(round((completed_count / total_count) * 100)) if total_count else 0

    cards = [
        {
            "label": "Tracked Actions",
            "value": f"{total_count:,}",
            "support": scope_label,
            "icon": "fact_check",
        },
        {
            "label": "Active Follow-Ups",
            "value": f"{active_count:,}",
            "support": "Planned or in progress",
            "icon": "pending_actions",
        },
        {
            "label": "Improvement Rate",
            "value": f"{improved_rate}%",
            "support": f"Across {reviewed_count:,} reviewed outcomes",
            "icon": "trending_up",
        },
        {
            "label": "Overdue Reviews",
            "value": f"{overdue_count:,}",
            "support": f"Completion rate {completion_rate}%",
            "icon": "schedule_send",
        },
    ]

    outcome_breakdown = []
    for outcome_value, outcome_label in InterventionRecord.Outcome.choices:
        count = queryset.filter(outcome=outcome_value).count()
        percent = int(round((count / total_count) * 100)) if total_count else 0
        outcome_breakdown.append(
            {
                "value": outcome_value,
                "label": outcome_label,
                "count": count,
                "percent": percent,
                "style": InterventionRecord.OUTCOME_STYLE_MAP.get(
                    outcome_value,
                    "bg-surface-container text-on-surface-variant",
                ),
            }
        )

    category_breakdown = []
    category_counts = queryset.values("category").annotate(count=Count("id")).order_by("-count", "category")
    for row in category_counts:
        category_value = row["category"]
        category_label = InterventionRecord.Category(category_value).label
        percent = int(round((row["count"] / total_count) * 100)) if total_count else 0
        category_breakdown.append(
            {
                "value": category_value,
                "label": category_label,
                "count": row["count"],
                "percent": percent,
            }
        )

    recent_records = list(queryset.order_by("-updated_at", "-created_at")[:5])
    for record in recent_records:
        record.scope_name = record.user.username

    return {
        "cards": cards,
        "outcome_breakdown": outcome_breakdown,
        "category_breakdown": category_breakdown,
        "recent_records": recent_records,
        "scope_label": scope_label,
        "total_count": total_count,
    }


def build_intervention_history_context(user, raw_filters=None):
    raw_filters = raw_filters or {}
    include_all_for_admin = get_user_role(user) == "admin"
    base_queryset = get_intervention_queryset(user, include_all_for_admin=include_all_for_admin)
    from .forms import InterventionHistoryFilterForm

    filter_form = InterventionHistoryFilterForm(raw_filters or None)

    queryset = base_queryset
    if filter_form.is_valid():
        cleaned_data = filter_form.cleaned_data
        if cleaned_data.get("student_id"):
            queryset = queryset.filter(student_id=cleaned_data["student_id"])
        if cleaned_data.get("category"):
            queryset = queryset.filter(category=cleaned_data["category"])
        if cleaned_data.get("severity"):
            queryset = queryset.filter(severity=cleaned_data["severity"])
        if cleaned_data.get("status"):
            queryset = queryset.filter(status=cleaned_data["status"])
        if cleaned_data.get("outcome"):
            queryset = queryset.filter(outcome=cleaned_data["outcome"])
        if cleaned_data.get("target_feature"):
            queryset = queryset.filter(target_feature=cleaned_data["target_feature"])
        if cleaned_data.get("date_from"):
            queryset = queryset.filter(updated_at__date__gte=cleaned_data["date_from"])
        if cleaned_data.get("date_to"):
            queryset = queryset.filter(updated_at__date__lte=cleaned_data["date_to"])

    records = list(queryset.order_by("-updated_at", "-created_at"))
    for record in records:
        record.scope_name = record.user.username

    scope_label = "All lecturer intervention records" if include_all_for_admin else "Your lecturer intervention records"
    analytics = build_intervention_analytics(queryset, scope_label)

    return {
        "filter_form": filter_form,
        "records": records,
        "record_count": len(records),
        "intervention_analytics": analytics,
        "scope_label": scope_label,
        "is_admin_scope": include_all_for_admin,
    }


def normalize_numeric_columns(dataframe, column_names):
    normalized = dataframe.copy()
    invalid_columns = []

    for column_name in column_names:
        if column_name not in normalized.columns:
            continue

        normalized[column_name] = pd.to_numeric(normalized[column_name], errors="coerce")
        if normalized[column_name].isna().any():
            invalid_columns.append(column_name)

    if invalid_columns:
        return None, invalid_columns

    if "StudentID" in normalized.columns:
        normalized["StudentID"] = normalized["StudentID"].astype(int)
    if "GradeClass" in normalized.columns:
        normalized["GradeClass"] = normalized["GradeClass"].astype(int)

    return normalized, []


def validate_dataset_frame(dataframe, required_columns):
    missing_columns = [
        column_name
        for column_name in required_columns
        if column_name not in dataframe.columns
    ]
    if missing_columns:
        return None, ["Missing required columns: " + ", ".join(missing_columns)]

    numeric_frame, invalid_columns = normalize_numeric_columns(
        dataframe,
        [column_name for column_name in required_columns if column_name in NUMERIC_DATASET_COLUMNS],
    )
    if invalid_columns:
        return None, [
            "The dataset contains blank or invalid numeric values in: "
            + ", ".join(invalid_columns)
        ]

    return numeric_frame, []


def get_active_dataset_record(user):
    if user is None or not getattr(user, "is_authenticated", False):
        return None

    return LecturerDataset.objects.filter(user=user, is_active=True).first()


def load_active_user_dataset(user, dataset_record=None):
    dataset_record = dataset_record or get_active_dataset_record(user)
    if dataset_record is None:
        return pd.DataFrame()

    try:
        dataset = dataset_record.load_dataframe()
    except Exception:
        return pd.DataFrame()

    normalized_dataset, errors = validate_dataset_frame(dataset, DATASET_REQUIRED_COLUMNS)
    if errors:
        return pd.DataFrame()

    return normalized_dataset


@lru_cache(maxsize=1)
def get_dataset():
    if not DATASET_PATH.exists():
        return pd.DataFrame()

    dataset = pd.read_csv(DATASET_PATH)
    if "GradeClass" in dataset.columns:
        dataset["GradeClass"] = pd.to_numeric(
            dataset["GradeClass"], errors="coerce"
        ).fillna(4).astype(int)
    return dataset


@lru_cache(maxsize=1)
def get_model():
    dataset = get_dataset()
    model = StudentPerformanceModel.load(MODEL_PATH)

    if model is not None or dataset.empty:
        return model

    if not set(StudentPerformanceModel.FEATURES).issubset(dataset.columns):
        return None

    trained_model = StudentPerformanceModel()
    trained_model.train(dataset[StudentPerformanceModel.FEATURES], dataset["GradeClass"])
    trained_model.save(MODEL_PATH)
    return trained_model


def get_form_defaults(user=None):
    if user is not None:
        dataset = load_active_user_dataset(user)
    else:
        dataset = get_dataset()

    if dataset.empty:
        return DEFAULT_FORM_VALUES.copy()

    return {
        "Age": int(round(dataset["Age"].median())),
        "Gender": int(dataset["Gender"].mode().iat[0]),
        "ParentalEducation": int(dataset["ParentalEducation"].mode().iat[0]),
        "StudyTimeWeekly": round(float(dataset["StudyTimeWeekly"].median()), 1),
        "Absences": int(round(dataset["Absences"].median())),
        "ParentalSupport": int(dataset["ParentalSupport"].mode().iat[0]),
        "Extracurricular": int(dataset["Extracurricular"].mode().iat[0]),
        "Sports": int(dataset["Sports"].mode().iat[0]),
        "Music": int(dataset["Music"].mode().iat[0]),
        "Volunteering": int(dataset["Volunteering"].mode().iat[0]),
    }


def coerce_feature_values(raw_values):
    return {
        "Age": int(raw_values.get("Age", 17)),
        "Gender": int(raw_values.get("Gender", 0)),
        "ParentalEducation": int(raw_values.get("ParentalEducation", 2)),
        "StudyTimeWeekly": float(raw_values.get("StudyTimeWeekly", 10)),
        "Absences": int(raw_values.get("Absences", 5)),
        "ParentalSupport": int(raw_values.get("ParentalSupport", 2)),
        "Extracurricular": int(raw_values.get("Extracurricular", 0)),
        "Sports": int(raw_values.get("Sports", 0)),
        "Music": int(raw_values.get("Music", 0)),
        "Volunteering": int(raw_values.get("Volunteering", 0)),
    }


def grade_label_from_class(grade_class):
    return StudentPerformanceModel.GRADE_MAP.get(int(grade_class), "Unknown")


def grade_class_from_label(grade_label):
    reverse_map = {label: grade_class for grade_class, label in StudentPerformanceModel.GRADE_MAP.items()}
    return reverse_map.get(grade_label, 2)


def normalize_period_column_name(column_name):
    return re.sub(r"[^a-z0-9]", "", str(column_name).lower())


def period_option_sort_key(value):
    text = str(value).strip()
    numbers = tuple(int(match) for match in re.findall(r"\d+", text))
    if numbers:
        return (0, numbers, text.lower())
    return (1, text.lower())


def detect_period_dimensions(dataset):
    dimensions = []
    if dataset.empty:
        return dimensions

    for column_name in dataset.columns:
        config = PERIOD_DIMENSION_CONFIG.get(normalize_period_column_name(column_name))
        if config is None:
            continue

        series = dataset[column_name]
        if config["type"] == "date":
            parsed_dates = pd.to_datetime(series, errors="coerce")
            if not parsed_dates.notna().any():
                continue

            dimensions.append(
                {
                    "column": column_name,
                    "label": config["label"],
                    "type": "date",
                    "min": parsed_dates.min().date().isoformat(),
                    "max": parsed_dates.max().date().isoformat(),
                }
            )
            continue

        options = [
            value
            for value in series.dropna().astype(str).str.strip().unique().tolist()
            if value
        ]
        if len(options) <= 1:
            continue

        dimensions.append(
            {
                "column": column_name,
                "label": config["label"],
                "type": "categorical",
                "options": sorted(options, key=period_option_sort_key),
            }
        )

    return dimensions


def count_high_risk_students(dataset):
    if dataset.empty:
        return 0

    count = 0
    for record in dataset.to_dict("records"):
        predicted_grade = grade_label_from_class(record.get("GradeClass", 4))
        risk_score = compute_risk_score(record, predicted_grade)
        if risk_label_from_score(risk_score) == "High Risk":
            count += 1
    return count


def build_metric_delta(label, current_value, previous_value, decimals=0, lower_is_better=False):
    delta = current_value - previous_value
    if decimals:
        current_display = f"{current_value:.{decimals}f}"
        previous_display = f"{previous_value:.{decimals}f}"
        delta_display = f"{delta:+.{decimals}f}"
    else:
        current_display = f"{int(round(current_value)):,}"
        previous_display = f"{int(round(previous_value)):,}"
        delta_display = f"{int(round(delta)):+,}"

    if delta == 0:
        tone = "neutral"
    elif (delta < 0 and lower_is_better) or (delta > 0 and not lower_is_better):
        tone = "positive"
    else:
        tone = "negative"

    return {
        "label": label,
        "current_display": current_display,
        "previous_display": previous_display,
        "delta_display": delta_display,
        "tone": tone,
    }


def build_period_filter_context(dataset, raw_filters=None):
    raw_filters = raw_filters or {}
    dimensions = detect_period_dimensions(dataset)

    context = {
        "available": bool(dimensions),
        "dimensions": dimensions,
        "selected_dimension": "",
        "selected_type": "",
        "value_options": [],
        "selected_value": "",
        "compare_options": [],
        "selected_compare_value": "",
        "date_from": "",
        "date_to": "",
        "min_date": "",
        "max_date": "",
        "compare_previous": False,
        "active_label": "All records",
        "helper_text": (
            "Add AcademicSession, Semester, Intake, AcademicYear, or RecordDate columns to unlock period filters."
            if not dimensions
            else "Filter the dashboard by academic period and compare current versus previous periods."
        ),
    }

    if not dimensions:
        return context, dataset, None, None

    selected_dimension = raw_filters.get("period_dimension", "")
    selected_config = next(
        (item for item in dimensions if item["column"] == selected_dimension),
        dimensions[0],
    )
    context["selected_dimension"] = selected_config["column"]
    context["selected_type"] = selected_config["type"]

    column_name = selected_config["column"]
    comparison_label = None

    if selected_config["type"] == "categorical":
        value_options = selected_config["options"]
        selected_value = raw_filters.get("period_value", "")
        if selected_value not in value_options:
            selected_value = value_options[0]

        compare_options = [value for value in value_options if value != selected_value]
        selected_compare_value = raw_filters.get("compare_value", "")
        if selected_compare_value not in compare_options:
            selected_compare_value = ""

        normalized_series = dataset[column_name].astype(str).str.strip()
        filtered_dataset = dataset[normalized_series == selected_value].copy()
        comparison_dataset = (
            dataset[normalized_series == selected_compare_value].copy()
            if selected_compare_value
            else None
        )

        context.update(
            {
                "value_options": value_options,
                "selected_value": selected_value,
                "compare_options": compare_options,
                "selected_compare_value": selected_compare_value,
                "active_label": f"{selected_config['label']}: {selected_value}",
            }
        )
        comparison_label = selected_compare_value or None
        return context, filtered_dataset, comparison_dataset, comparison_label

    parsed_dates = pd.to_datetime(dataset[column_name], errors="coerce")
    valid_dates = parsed_dates.dropna()
    min_date = valid_dates.min().date().isoformat()
    max_date = valid_dates.max().date().isoformat()
    requested_from = raw_filters.get("date_from") or min_date
    requested_to = raw_filters.get("date_to") or max_date
    start_date = pd.to_datetime(requested_from, errors="coerce")
    end_date = pd.to_datetime(requested_to, errors="coerce")

    if pd.isna(start_date):
        start_date = pd.to_datetime(min_date)
    if pd.isna(end_date):
        end_date = pd.to_datetime(max_date)
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    compare_previous = str(raw_filters.get("compare_previous", "")).lower() in {
        "1",
        "true",
        "on",
        "yes",
    }
    filtered_dataset = dataset[parsed_dates.between(start_date, end_date, inclusive="both")].copy()
    comparison_dataset = None

    if compare_previous:
        span_days = max((end_date - start_date).days, 0)
        previous_end = start_date - pd.Timedelta(days=1)
        previous_start = previous_end - pd.Timedelta(days=span_days)
        comparison_dataset = dataset[
            parsed_dates.between(previous_start, previous_end, inclusive="both")
        ].copy()
        comparison_label = f"{previous_start.date().isoformat()} to {previous_end.date().isoformat()}"

    context.update(
        {
            "date_from": start_date.date().isoformat(),
            "date_to": end_date.date().isoformat(),
            "min_date": min_date,
            "max_date": max_date,
            "compare_previous": compare_previous,
            "active_label": f"{selected_config['label']}: {start_date.date().isoformat()} to {end_date.date().isoformat()}",
        }
    )
    return context, filtered_dataset, comparison_dataset, comparison_label


def build_period_comparison(context, comparison_dataset, comparison_label):
    if comparison_dataset is None or comparison_dataset.empty:
        return None

    current_dataset = context["filtered_dataset"]
    current_gpa = float(current_dataset["GPA"].mean()) if not current_dataset.empty else 0.0
    previous_gpa = float(comparison_dataset["GPA"].mean()) if not comparison_dataset.empty else 0.0
    current_absences = float(current_dataset["Absences"].mean()) if not current_dataset.empty else 0.0
    previous_absences = float(comparison_dataset["Absences"].mean()) if not comparison_dataset.empty else 0.0
    current_high_risk = count_high_risk_students(current_dataset)
    previous_high_risk = count_high_risk_students(comparison_dataset)

    return {
        "current_label": context["period_filters"]["active_label"],
        "comparison_label": comparison_label,
        "cards": [
            build_metric_delta("Students", len(current_dataset), len(comparison_dataset)),
            build_metric_delta("Average GPA", current_gpa, previous_gpa, decimals=2),
            build_metric_delta(
                "High Risk Students",
                current_high_risk,
                previous_high_risk,
                lower_is_better=True,
            ),
            build_metric_delta(
                "Average Absences",
                current_absences,
                previous_absences,
                decimals=1,
                lower_is_better=True,
            ),
        ],
    }


def compute_risk_score(values, predicted_grade=None):
    absences_ratio = min(float(values.get("Absences", 0)) / 30.0, 1.0)
    study_ratio = 1.0 - min(float(values.get("StudyTimeWeekly", 0)) / 20.0, 1.0)
    support_ratio = 1.0 - min(float(values.get("ParentalSupport", 0)) / 4.0, 1.0)
    gpa_ratio = 0.0

    if "GPA" in values and values.get("GPA") is not None:
        gpa_ratio = 1.0 - min(max(float(values.get("GPA", 0)), 0.0) / 4.0, 1.0)

    activity_count = sum(
        int(values.get(feature, 0))
        for feature in ["Extracurricular", "Sports", "Music", "Volunteering"]
    )
    activity_penalty = max(0.0, 1.0 - (activity_count * 0.2))

    score = (
        (absences_ratio * 34)
        + (study_ratio * 24)
        + (support_ratio * 16)
        + (gpa_ratio * 18)
        + (activity_penalty * 8)
    )

    if predicted_grade == "F":
        score += 18
    elif predicted_grade == "D":
        score += 10
    elif predicted_grade == "C":
        score += 4

    return int(max(0, min(round(score), 100)))


def risk_label_from_score(score):
    if score >= 70:
        return "High Risk"
    if score >= 45:
        return "Moderate"
    return "Low Risk"


def estimate_confidence(model, feature_frame, risk_score):
    if model is not None and hasattr(model.model, "predict_proba"):
        scaled_values = model.scaler.transform(feature_frame)
        probabilities = model.model.predict_proba(scaled_values)[0]
        return int(round(float(probabilities.max()) * 100))

    distance_from_border = abs(risk_score - 50)
    return int(max(60, min(94, 70 + (distance_from_border / 2))))


def format_feature_value(feature_name, value):
    if value is None:
        return "Unknown"
    if feature_name == "StudyTimeWeekly":
        return f"{float(value):.1f} hours/week"
    if feature_name in {"Gender", "Extracurricular", "Sports", "Music", "Volunteering"}:
        if feature_name == "Gender":
            return GENDER_LABELS.get(int(value), str(value))
        return BOOLEAN_LABELS.get(int(value), str(value))
    if feature_name == "ParentalSupport":
        return SUPPORT_LABELS[int(value)]
    if feature_name == "ParentalEducation":
        return EDUCATION_LABELS[int(value)]
    if feature_name in {"Age", "Absences"}:
        return str(int(value))
    return str(value)


@lru_cache(maxsize=1)
def get_shap_background_frame():
    dataset = get_dataset()
    if dataset.empty or not set(StudentPerformanceModel.FEATURES).issubset(dataset.columns):
        return pd.DataFrame([DEFAULT_FORM_VALUES], columns=StudentPerformanceModel.FEATURES)

    background = dataset[StudentPerformanceModel.FEATURES].copy()
    if len(background) > 80:
        return background.sample(80, random_state=42)
    return background


@lru_cache(maxsize=1)
def get_shap_explainer_bundle():
    model = get_model()
    if model is None or model.model is None:
        return None

    try:
        import shap
    except ImportError:
        return None

    estimator = model.model
    try:
        if hasattr(estimator, "feature_importances_"):
            return {
                "type": "tree",
                "explainer": shap.TreeExplainer(estimator),
            }

        if hasattr(estimator, "coef_"):
            background = get_shap_background_frame()
            scaled_background = model.scaler.transform(background)
            return {
                "type": "linear",
                "explainer": shap.LinearExplainer(estimator, scaled_background),
            }
    except Exception:
        return None

    return None


def extract_target_shap_values(shap_output, predicted_class, feature_count):
    values = getattr(shap_output, "values", shap_output)

    if isinstance(values, list):
        class_index = max(0, min(predicted_class, len(values) - 1))
        return np.asarray(values[class_index])[0]

    values = np.asarray(values)
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        return values[0]
    if values.ndim == 3:
        if values.shape[0] == 1 and values.shape[1] == feature_count:
            class_index = max(0, min(predicted_class, values.shape[2] - 1))
            return values[0, :, class_index]
        if values.shape[0] == 1 and values.shape[2] == feature_count:
            class_index = max(0, min(predicted_class, values.shape[1] - 1))
            return values[0, class_index, :]
        if values.shape[1] == 1 and values.shape[2] == feature_count:
            class_index = max(0, min(predicted_class, values.shape[0] - 1))
            return values[class_index, 0, :]

    raise ValueError("Unsupported SHAP output shape")


def infer_value_based_tone(feature_name, value):
    if feature_name == "Absences":
        if int(value) >= 8:
            return "negative"
        return "positive"
    if feature_name == "StudyTimeWeekly":
        if float(value) < 8:
            return "negative"
        return "positive"
    if feature_name == "ParentalSupport":
        if int(value) <= 1:
            return "negative"
        return "positive"
    if feature_name == "ParentalEducation":
        if int(value) <= 1:
            return "negative"
        return "positive"
    if feature_name in ACTIVITY_FEATURES:
        return "positive" if int(value) else "negative"
    return None


def infer_factor_tone(feature_name, value, shap_value, predicted_class):
    if predicted_class >= 3:
        return "negative" if shap_value > 0 else "positive"
    if predicted_class <= 1:
        return "positive" if shap_value > 0 else "negative"

    heuristic_tone = infer_value_based_tone(feature_name, value)
    if heuristic_tone is not None:
        return heuristic_tone
    return "negative" if shap_value > 0 else "positive"


def build_shap_factor_description(feature_name, value, tone, predicted_grade):
    value_display = format_feature_value(feature_name, value)
    feature_label = FEATURE_LABELS.get(feature_name, feature_name)

    if feature_name == "Absences":
        if tone == "negative":
            return f"{value_display} recorded absences were one of the strongest SHAP drivers behind this student's {predicted_grade} outlook."
        return f"{value_display} recorded absences helped stabilize the current {predicted_grade} outlook."

    if feature_name == "StudyTimeWeekly":
        if tone == "negative":
            return f"{value_display} of study time was ranked by SHAP as a major drag on the current {predicted_grade} outlook."
        return f"{value_display} of study time supported the current {predicted_grade} outlook."

    if feature_name == "ParentalSupport":
        if tone == "negative":
            return f"Parental support is currently {value_display}, and SHAP flagged it as a meaningful risk driver."
        return f"Parental support is {value_display}, which helped the current prediction stay stable."

    if feature_name == "ParentalEducation":
        if tone == "negative":
            return f"Parental education is recorded as {value_display}, and it contributed to the present {predicted_grade} outlook."
        return f"Parental education at {value_display} supported the current prediction."

    if feature_name in ACTIVITY_FEATURES:
        if tone == "negative":
            return f"{feature_label} is currently {value_display}, and SHAP treated that low engagement signal as one of the strongest local drivers."
        return f"{feature_label} is currently {value_display}, which reinforced the current prediction."

    if feature_name == "Gender":
        return f"{feature_label} influenced the model locally, but it should not be treated as an intervention target on its own."

    if feature_name == "Age":
        return f"{feature_label} influenced the model locally, but intervention planning should focus on attendance, study habits, and support instead."

    if tone == "negative":
        return f"{feature_label} ({value_display}) was one of the strongest SHAP drivers behind the current {predicted_grade} outlook."
    return f"{feature_label} ({value_display}) supported the current {predicted_grade} outlook."


def build_shap_feature_factors(values, predicted_class, predicted_grade, model, feature_frame):
    explainer_bundle = get_shap_explainer_bundle()
    if explainer_bundle is None:
        return []

    scaled_values = model.scaler.transform(feature_frame)
    try:
        shap_output = explainer_bundle["explainer"].shap_values(scaled_values)
        target_values = extract_target_shap_values(
            shap_output,
            predicted_class,
            len(StudentPerformanceModel.FEATURES),
        )
    except Exception:
        return []

    absolute_values = np.abs(target_values)
    total_absolute = float(absolute_values.sum())
    if total_absolute <= 0:
        return []

    factors = []
    ranked_indexes = np.argsort(absolute_values)[::-1][:3]
    for feature_index in ranked_indexes:
        feature_name = StudentPerformanceModel.FEATURES[feature_index]
        feature_value = values.get(feature_name)
        tone = infer_factor_tone(
            feature_name,
            feature_value,
            float(target_values[feature_index]),
            predicted_class,
        )
        impact_share = max(1, int(round((float(absolute_values[feature_index]) / total_absolute) * 100)))
        factors.append(
            {
                "feature": feature_name,
                "title": FEATURE_LABELS.get(feature_name, feature_name),
                "impact": impact_share if tone == "positive" else -impact_share,
                "impact_share": impact_share,
                "tone": tone,
                "description": build_shap_factor_description(
                    feature_name,
                    feature_value,
                    tone,
                    predicted_grade,
                ),
                "value_display": format_feature_value(feature_name, feature_value),
                "shap_value": round(float(target_values[feature_index]), 6),
            }
        )

    return factors


def build_heuristic_feature_factors(values):
    absences = int(values.get("Absences", 0))
    study_time = float(values.get("StudyTimeWeekly", 0))
    parental_support = int(values.get("ParentalSupport", 0))
    activity_count = sum(
        int(values.get(feature, 0))
        for feature in ["Extracurricular", "Sports", "Music", "Volunteering"]
    )

    factors = []

    if absences >= 12:
        factors.append(
            {
                "feature": "Absences",
                "title": "Absence pattern",
                "impact": -32,
                "impact_share": 32,
                "tone": "negative",
                "description": "Frequent absences are the strongest warning signal in the current profile.",
            }
        )
    elif absences <= 4:
        factors.append(
            {
                "feature": "Absences",
                "title": "Attendance consistency",
                "impact": 22,
                "impact_share": 22,
                "tone": "positive",
                "description": "Low absence count suggests stable participation and better course continuity.",
            }
        )

    if study_time < 6:
        factors.append(
            {
                "feature": "StudyTimeWeekly",
                "title": "Weekly study time",
                "impact": -21,
                "impact_share": 21,
                "tone": "negative",
                "description": "Low study hours increase the chance of weaker GPA outcomes and missed revision.",
            }
        )
    elif study_time >= 12:
        factors.append(
            {
                "feature": "StudyTimeWeekly",
                "title": "Weekly study time",
                "impact": 18,
                "impact_share": 18,
                "tone": "positive",
                "description": "Consistent study hours usually correlate with stronger grade stability in the dataset.",
            }
        )

    if parental_support <= 1:
        factors.append(
            {
                "feature": "ParentalSupport",
                "title": "Support environment",
                "impact": -12,
                "impact_share": 12,
                "tone": "negative",
                "description": "Lower parental support can reduce resilience when attendance or workload starts slipping.",
            }
        )
    elif parental_support >= 3:
        factors.append(
            {
                "feature": "ParentalSupport",
                "title": "Support environment",
                "impact": 10,
                "impact_share": 10,
                "tone": "positive",
                "description": "Higher support levels often align with stronger intervention response and routine stability.",
            }
        )

    if activity_count >= 2:
        factors.append(
            {
                "feature": "Extracurricular",
                "title": "Co-curricular engagement",
                "impact": 8,
                "impact_share": 8,
                "tone": "positive",
                "description": "Balanced activity participation can reflect stronger structure and social belonging.",
            }
        )

    if not factors:
        factors.append(
            {
                "feature": "",
                "title": "Balanced profile",
                "impact": 6,
                "impact_share": 6,
                "tone": "neutral",
                "description": "No single feature is dominating the prediction, so the result depends on the overall feature mix.",
            }
        )

    factors.sort(key=lambda item: abs(item["impact"]), reverse=True)
    return factors[:3]


def build_feature_factors(values, predicted_class=None, predicted_grade=None, model=None, feature_frame=None):
    if model is not None and predicted_class is not None and predicted_grade is not None and feature_frame is not None:
        shap_factors = build_shap_feature_factors(
            values,
            predicted_class,
            predicted_grade,
            model,
            feature_frame,
        )
        if shap_factors:
            return shap_factors

    return build_heuristic_feature_factors(values)


def build_recommendation_from_factor(factor, values, risk_label, predicted_grade):
    feature_name = factor.get("feature") or ""
    impact_share = factor.get("impact_share", abs(int(factor.get("impact", 0))))
    shap_value = factor.get("shap_value")

    if feature_name == "Absences" and int(values.get("Absences", 0)) >= 8:
        value = int(values.get("Absences", 0))
        severity = (
            InterventionRecord.Severity.URGENT
            if risk_label == "High Risk" or value >= 10
            else InterventionRecord.Severity.RECOMMENDED
        )
        return {
            "title": "Attendance intervention",
            "description": f"This student has {value} absences, and SHAP ranked attendance as a top driver. Schedule a same-week check-in and confirm what is blocking attendance.",
            "icon": "event_busy",
            "category": InterventionRecord.Category.ATTENDANCE,
            "category_label": InterventionRecord.Category.ATTENDANCE.label,
            "severity": severity,
            "severity_label": InterventionRecord.Severity(severity).label,
            "severity_style": RECOMMENDATION_SEVERITY_STYLE_MAP[severity],
            "target_feature": feature_name,
            "target_feature_label": FEATURE_LABELS[feature_name],
            "feature_value": value,
            "feature_value_display": format_feature_value(feature_name, value),
            "shap_value": shap_value,
            "impact_share": impact_share,
        }

    if feature_name == "StudyTimeWeekly" and float(values.get("StudyTimeWeekly", 0)) < 8:
        value = float(values.get("StudyTimeWeekly", 0))
        severity = (
            InterventionRecord.Severity.URGENT
            if risk_label == "High Risk" and value < 6
            else InterventionRecord.Severity.RECOMMENDED
        )
        return {
            "title": "Study plan reset",
            "description": f"Study time is only {value:.1f} hours per week, and SHAP placed it near the top of the local explanation. Set two fixed revision blocks and review them at the next check-in.",
            "icon": "schedule",
            "category": InterventionRecord.Category.STUDY,
            "category_label": InterventionRecord.Category.STUDY.label,
            "severity": severity,
            "severity_label": InterventionRecord.Severity(severity).label,
            "severity_style": RECOMMENDATION_SEVERITY_STYLE_MAP[severity],
            "target_feature": feature_name,
            "target_feature_label": FEATURE_LABELS[feature_name],
            "feature_value": value,
            "feature_value_display": format_feature_value(feature_name, value),
            "shap_value": shap_value,
            "impact_share": impact_share,
        }

    if feature_name == "ParentalSupport" and int(values.get("ParentalSupport", 0)) <= 1:
        value = int(values.get("ParentalSupport", 0))
        severity = (
            InterventionRecord.Severity.URGENT
            if risk_label == "High Risk"
            else InterventionRecord.Severity.RECOMMENDED
        )
        return {
            "title": "Advisor follow-up",
            "description": f"Parental support is currently {SUPPORT_LABELS[value]}, and SHAP identified it as a notable local risk signal. Escalate to the advisor for closer monitoring and support planning.",
            "icon": "support_agent",
            "category": InterventionRecord.Category.SUPPORT,
            "category_label": InterventionRecord.Category.SUPPORT.label,
            "severity": severity,
            "severity_label": InterventionRecord.Severity(severity).label,
            "severity_style": RECOMMENDATION_SEVERITY_STYLE_MAP[severity],
            "target_feature": feature_name,
            "target_feature_label": FEATURE_LABELS[feature_name],
            "feature_value": value,
            "feature_value_display": format_feature_value(feature_name, value),
            "shap_value": shap_value,
            "impact_share": impact_share,
        }

    if feature_name == "ParentalEducation" and int(values.get("ParentalEducation", 0)) <= 1:
        value = int(values.get("ParentalEducation", 0))
        severity = InterventionRecord.Severity.RECOMMENDED
        return {
            "title": "Family support briefing",
            "description": f"Parental education is recorded as {EDUCATION_LABELS[value]}, and SHAP still ranked it as an important local factor. Share a simple support checklist with the household and advisor.",
            "icon": "family_home",
            "category": InterventionRecord.Category.SUPPORT,
            "category_label": InterventionRecord.Category.SUPPORT.label,
            "severity": severity,
            "severity_label": InterventionRecord.Severity(severity).label,
            "severity_style": RECOMMENDATION_SEVERITY_STYLE_MAP[severity],
            "target_feature": feature_name,
            "target_feature_label": FEATURE_LABELS[feature_name],
            "feature_value": value,
            "feature_value_display": format_feature_value(feature_name, value),
            "shap_value": shap_value,
            "impact_share": impact_share,
        }

    if feature_name in ACTIVITY_FEATURES and int(values.get(feature_name, 0)) == 0:
        feature_label = FEATURE_LABELS[feature_name]
        severity = (
            InterventionRecord.Severity.RECOMMENDED
            if risk_label in {"High Risk", "Moderate"}
            else InterventionRecord.Severity.OPTIONAL
        )
        return {
            "title": "Engagement activation",
            "description": f"{feature_label} is currently marked No, and SHAP still ranked it as one of the main local drivers. Match the student to one structured engagement option before the next review.",
            "icon": "groups",
            "category": InterventionRecord.Category.ENGAGEMENT,
            "category_label": InterventionRecord.Category.ENGAGEMENT.label,
            "severity": severity,
            "severity_label": InterventionRecord.Severity(severity).label,
            "severity_style": RECOMMENDATION_SEVERITY_STYLE_MAP[severity],
            "target_feature": feature_name,
            "target_feature_label": feature_label,
            "feature_value": int(values.get(feature_name, 0)),
            "feature_value_display": format_feature_value(feature_name, values.get(feature_name, 0)),
            "shap_value": shap_value,
            "impact_share": impact_share,
        }

    return None


def build_recommendations(values, risk_label, factors=None, predicted_grade=None):
    recommendations = []
    seen_categories = set()

    for factor in factors or []:
        recommendation = build_recommendation_from_factor(
            factor,
            values,
            risk_label,
            predicted_grade or "C",
        )
        if recommendation is None or recommendation["category"] in seen_categories:
            continue
        recommendations.append(recommendation)
        seen_categories.add(recommendation["category"])
        if len(recommendations) >= 3:
            return recommendations

    if risk_label == "Low Risk":
        recommendations.append(
            {
                "title": "Stretch goal planning",
                "description": f"The current {predicted_grade or 'strong'} outlook is stable. Consider advanced coursework, peer mentoring, or a higher-level academic target instead of a recovery plan.",
                "icon": "trending_up",
                "category": InterventionRecord.Category.ACHIEVEMENT,
                "category_label": InterventionRecord.Category.ACHIEVEMENT.label,
                "severity": InterventionRecord.Severity.OPTIONAL,
                "severity_label": InterventionRecord.Severity.OPTIONAL.label,
                "severity_style": RECOMMENDATION_SEVERITY_STYLE_MAP[InterventionRecord.Severity.OPTIONAL],
                "target_feature": "",
                "target_feature_label": "Growth",
                "feature_value": None,
                "feature_value_display": "",
                "shap_value": None,
                "impact_share": 0,
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "title": "Routine monitoring",
                "description": "Keep this student on the standard review cadence, record the next follow-up, and compare the outcome after the next assessment cycle.",
                "icon": "monitoring",
                "category": InterventionRecord.Category.MONITORING,
                "category_label": InterventionRecord.Category.MONITORING.label,
                "severity": InterventionRecord.Severity.RECOMMENDED,
                "severity_label": InterventionRecord.Severity.RECOMMENDED.label,
                "severity_style": RECOMMENDATION_SEVERITY_STYLE_MAP[InterventionRecord.Severity.RECOMMENDED],
                "target_feature": "",
                "target_feature_label": "General",
                "feature_value": None,
                "feature_value_display": "",
                "shap_value": None,
                "impact_share": 0,
            }
        )

    return recommendations[:3]


def build_prediction_context(feature_values):
    model = get_model()
    normalized_values = coerce_feature_values(feature_values)
    feature_frame = pd.DataFrame([normalized_values], columns=StudentPerformanceModel.FEATURES)

    if model is not None:
        predicted_class = int(model.predict(feature_frame)[0])
        predicted_grade = grade_label_from_class(predicted_class)
    else:
        if normalized_values["Absences"] >= 12 or normalized_values["StudyTimeWeekly"] < 6:
            predicted_grade = "D"
        elif normalized_values["StudyTimeWeekly"] >= 12 and normalized_values["Absences"] <= 4:
            predicted_grade = "A"
        else:
            predicted_grade = "C"
        predicted_class = grade_class_from_label(predicted_grade)

    risk_score = compute_risk_score(normalized_values, predicted_grade)
    risk_label = risk_label_from_score(risk_score)
    confidence = estimate_confidence(model, feature_frame, risk_score)
    factors = build_feature_factors(
        normalized_values,
        predicted_class=predicted_class,
        predicted_grade=predicted_grade,
        model=model,
        feature_frame=feature_frame,
    )

    return {
        "predicted_grade": predicted_grade,
        "risk_score": risk_score,
        "risk_label": risk_label,
        "risk_style": RISK_STYLE_MAP[risk_label],
        "confidence": confidence,
        "factors": factors,
        "recommendations": build_recommendations(
            normalized_values,
            risk_label,
            factors=factors,
            predicted_grade=predicted_grade,
        ),
    }


def build_dashboard_context(raw_filters=None, user=None):
    if user is not None:
        dataset_record = get_active_dataset_record(user)
        dataset = load_active_user_dataset(user, dataset_record=dataset_record)
        dataset_info = build_dataset_info(dataset_record)
        if dataset_record is None:
            return empty_dashboard_context(
                empty_period_filters(
                    "Upload and confirm your personal dataset first. Period filters will appear after activation.",
                    active_label="No active dataset",
                ),
                "No personal dataset yet",
                "Upload a CSV through My Dataset, confirm the detected column mapping, and your private dashboard will activate immediately.",
                dataset_info=None,
            )
        if dataset.empty:
            return empty_dashboard_context(
                empty_period_filters(
                    "The active dataset could not be read with the saved column mapping. Reconfirm the mapping or upload a clean CSV.",
                    active_label="Dataset unavailable",
                ),
                "Dataset needs attention",
                "The active dataset could not be loaded. Open My Dataset and confirm the mapping again.",
                dataset_info=dataset_info,
            )
    else:
        dataset_record = None
        dataset = get_dataset()
        dataset_info = None

    model = get_model()
    intervention_analytics = None
    recent_interventions = []
    if user is not None and getattr(user, "is_authenticated", False):
        intervention_queryset = get_intervention_queryset(user)
        intervention_analytics = build_intervention_analytics(intervention_queryset, "Your tracked intervention records")
        recent_interventions = intervention_analytics["recent_records"]
    period_filters, filtered_dataset, comparison_dataset, comparison_label = build_period_filter_context(
        dataset,
        raw_filters,
    )

    if dataset.empty:
        return empty_dashboard_context(
            period_filters,
            "No dataset available",
            "Place your CSV dataset in the project root or use the Upload page so the dashboard can populate real metrics.",
            dataset_info=dataset_info,
        )

    if filtered_dataset.empty:
        return empty_dashboard_context(
            period_filters,
            "No records match the selected period",
            "Adjust the academic-period filters or clear the comparison selection to see results again.",
            dataset_info=dataset_info,
        )

    dataset = filtered_dataset

    grade_counts = dataset["GradeClass"].value_counts().sort_index()
    grade_chart = {
        "labels": [grade_label_from_class(grade_class) for grade_class in grade_counts.index],
        "counts": [int(count) for count in grade_counts.values],
    }

    risk_labels = []
    top_students = []
    for record in dataset.to_dict("records"):
        risk_score = compute_risk_score(record, grade_label_from_class(record["GradeClass"]))
        risk_label = risk_label_from_score(risk_score)
        risk_labels.append(risk_label)

        student_id = int(record.get("StudentID", 0))
        top_students.append(
            {
                "student_id": student_id,
                "display_name": f"Student {student_id}",
                "grade": grade_label_from_class(record["GradeClass"]),
                "gpa": round(float(record.get("GPA", 0)), 2),
                "risk_score": risk_score,
                "risk_label": risk_label,
                "risk_style": RISK_STYLE_MAP[risk_label],
                "absences": int(record.get("Absences", 0)),
            }
        )

    risk_series = pd.Series(risk_labels).value_counts()
    risk_chart = {
        "labels": ["High Risk", "Moderate", "Low Risk"],
        "counts": [
            int(risk_series.get("High Risk", 0)),
            int(risk_series.get("Moderate", 0)),
            int(risk_series.get("Low Risk", 0)),
        ],
    }

    scatter_sample = dataset.sample(min(len(dataset), 120), random_state=42)
    study_scatter = [
        {"x": round(float(row.StudyTimeWeekly), 2), "y": round(float(row.GPA), 2)}
        for row in scatter_sample.itertuples()
    ]
    absences_scatter = [
        {"x": int(row.Absences), "y": round(float(row.GPA), 2)}
        for row in scatter_sample.itertuples()
    ]

    if model is not None and model.feature_importance is not None:
        feature_insights = [
            {
                "name": row.feature,
                "value": round(float(row.importance) * 100, 1),
            }
            for row in model.feature_importance.head(4).itertuples()
        ]
    else:
        correlations = (
            dataset[StudentPerformanceModel.FEATURES + ["GPA"]]
            .corr(numeric_only=True)["GPA"]
            .drop("GPA")
            .abs()
            .sort_values(ascending=False)
            .head(4)
        )
        feature_insights = [
            {"name": name, "value": round(float(score) * 100, 1)}
            for name, score in correlations.items()
        ]

    summary_cards = [
        {
            "label": "Total Students",
            "value": f"{len(dataset):,}",
            "support": f"Within {period_filters['active_label'].lower()}",
            "icon": "groups",
            "tone": "neutral",
        },
        {
            "label": "Average GPA",
            "value": f"{dataset['GPA'].mean():.2f}",
            "support": f"For {period_filters['active_label'].lower()}",
            "icon": "school",
            "tone": "neutral",
        },
        {
            "label": "At-Risk Students",
            "value": f"{risk_chart['counts'][0]:,}",
            "support": f"High risk inside {period_filters['active_label'].lower()}",
            "icon": "warning",
            "tone": "danger",
        },
        {
            "label": "Best Model Score",
            "value": (
                f"{model.comparison_results[model.best_model_name]['f1_score']:.1%}"
                if model is not None
                else "Pending"
            ),
            "support": model.best_model_name if model is not None else "Train a model to unlock metrics",
            "icon": "auto_awesome",
            "tone": "accent",
        },
    ]

    top_students.sort(key=lambda item: item["risk_score"], reverse=True)

    interventions = []
    high_absence_count = int((dataset["Absences"] >= 10).sum())
    low_study_count = int((dataset["StudyTimeWeekly"] < 6).sum())
    high_risk_count = risk_chart["counts"][0]

    if high_absence_count:
        interventions.append(
            {
                "title": "Attendance alert batch",
                "description": f"{high_absence_count} students crossed the high-absence threshold and should receive follow-up.",
                "icon": "mail",
            }
        )
    if low_study_count:
        interventions.append(
            {
                "title": "Study-skills workshop",
                "description": f"{low_study_count} students show low weekly study time and may benefit from guided planning.",
                "icon": "menu_book",
            }
        )
    if high_risk_count:
        interventions.append(
            {
                "title": "Advisor escalation queue",
                "description": f"{high_risk_count} students are currently classified as high risk and should be reviewed first.",
                "icon": "campaign",
            }
        )

    model_snapshot = None
    if model is not None:
        model_snapshot = {
            "name": model.best_model_name,
            "f1_score": f"{model.comparison_results[model.best_model_name]['f1_score']:.1%}",
            "accuracy": f"{model.comparison_results[model.best_model_name]['accuracy']:.1%}",
            "dataset_rows": len(dataset),
        }

    return {
        "summary_cards": summary_cards,
        "grade_chart": grade_chart,
        "risk_chart": risk_chart,
        "study_scatter": study_scatter,
        "absences_scatter": absences_scatter,
        "top_students": top_students[:5],
        "feature_insights": feature_insights,
        "interventions": interventions,
        "model_snapshot": model_snapshot,
        "intervention_analytics": intervention_analytics,
        "recent_interventions": recent_interventions,
        "dataset_info": dataset_info,
        "period_filters": period_filters,
        "period_comparison": build_period_comparison(
            {
                "filtered_dataset": dataset,
                "period_filters": period_filters,
            },
            comparison_dataset,
            comparison_label,
        ),
        "dashboard_empty_title": "No dataset available",
        "dashboard_empty_message": "Place your CSV dataset in the project root or use the Upload page so the dashboard can populate real metrics.",
    }


def build_batch_context(uploaded_file, preferred_mapping=None):
    uploaded_file.seek(0)
    try:
        dataset = pd.read_csv(uploaded_file)
    except Exception as exc:
        return {"error": f"Could not read the uploaded CSV file: {exc}"}

    mapping_result = resolve_columns(
        dataset,
        canonical_columns=BATCH_REQUIRED_COLUMNS + BATCH_OPTIONAL_COLUMNS,
        preferred_mapping=preferred_mapping,
    )

    try:
        dataset = apply_column_mapping(dataset, mapping_result["auto_mapped"])
    except ValueError as exc:
        return {"error": str(exc)}

    missing_columns = [
        feature for feature in BATCH_REQUIRED_COLUMNS if feature not in dataset.columns
    ]
    if missing_columns:
        return {
            "error": "Missing required columns after auto-detection: " + ", ".join(missing_columns)
        }

    numeric_frame, invalid_columns = normalize_numeric_columns(
        dataset,
        BATCH_REQUIRED_COLUMNS + BATCH_OPTIONAL_COLUMNS,
    )
    if invalid_columns:
        return {
            "error": "The uploaded file contains blank or invalid numeric values in: " + ", ".join(invalid_columns)
        }

    model = get_model()
    features_only = numeric_frame[StudentPerformanceModel.FEATURES]

    if model is not None:
        predicted_classes = model.predict(features_only)
        predicted_grades = [grade_label_from_class(value) for value in predicted_classes]
    else:
        predicted_grades = []
        for record in features_only.to_dict("records"):
            if record["Absences"] >= 12 or record["StudyTimeWeekly"] < 6:
                predicted_grades.append("D")
            elif record["StudyTimeWeekly"] >= 12 and record["Absences"] <= 4:
                predicted_grades.append("A")
            else:
                predicted_grades.append("C")

    rows = []
    risk_counts = {"High Risk": 0, "Moderate": 0, "Low Risk": 0}
    grade_counts = {}

    for index, record in enumerate(numeric_frame.to_dict("records")):
        predicted_grade = predicted_grades[index]
        risk_score = compute_risk_score(record, predicted_grade)
        risk_label = risk_label_from_score(risk_score)
        risk_counts[risk_label] += 1
        grade_counts[predicted_grade] = grade_counts.get(predicted_grade, 0) + 1

        feature_frame = pd.DataFrame([coerce_feature_values(record)], columns=StudentPerformanceModel.FEATURES)
        rows.append(
            {
                "student_id": int(record.get("StudentID", index + 1)),
                "display_name": f"Student {int(record.get('StudentID', index + 1))}",
                "predicted_grade": predicted_grade,
                "risk_label": risk_label,
                "risk_style": RISK_STYLE_MAP[risk_label],
                "risk_score": risk_score,
                "confidence": estimate_confidence(model, feature_frame, risk_score),
            }
        )

    grade_distribution = [
        {"label": label, "count": count}
        for label, count in sorted(grade_counts.items(), key=lambda item: item[0])
    ]

    return {
        "file_name": uploaded_file.name,
        "processed_count": len(rows),
        "high_risk_count": risk_counts["High Risk"],
        "moderate_risk_count": risk_counts["Moderate"],
        "low_risk_count": risk_counts["Low Risk"],
        "grade_distribution": grade_distribution,
        "results": rows[:20],
    }


def get_student_context(student_id, user=None):
    if user is not None:
        dataset = load_active_user_dataset(user)
        if dataset.empty:
            raise StudentNotFoundError("Upload and confirm your personal dataset first.")
    else:
        dataset = get_dataset()

    if dataset.empty or "StudentID" not in dataset.columns:
        raise StudentNotFoundError("Student dataset is not available.")

    matches = dataset[dataset["StudentID"] == student_id]
    if matches.empty:
        raise StudentNotFoundError(f"Student {student_id} was not found in the dataset.")

    record = matches.iloc[0].to_dict()
    features = {feature: record[feature] for feature in StudentPerformanceModel.FEATURES}
    prediction = build_prediction_context(features)
    actual_grade = grade_label_from_class(record["GradeClass"])

    metrics = [
        {
            "label": "GPA",
            "value": f"{float(record.get('GPA', 0)):.2f}",
            "progress": int(min(max((float(record.get("GPA", 0)) / 4.0) * 100, 0), 100)),
            "tone": "positive",
        },
        {
            "label": "Study time",
            "value": f"{float(record.get('StudyTimeWeekly', 0)):.1f} hrs/week",
            "progress": int(min(max((float(record.get("StudyTimeWeekly", 0)) / 20.0) * 100, 0), 100)),
            "tone": "positive",
        },
        {
            "label": "Absences",
            "value": f"{int(record.get('Absences', 0))} days",
            "progress": int(min(max((int(record.get("Absences", 0)) / 30.0) * 100, 0), 100)),
            "tone": "negative",
        },
        {
            "label": "Parental support",
            "value": SUPPORT_LABELS[int(record.get("ParentalSupport", 0))],
            "progress": int(min(max((int(record.get("ParentalSupport", 0)) / 4.0) * 100, 0), 100)),
            "tone": "positive",
        },
    ]

    activities = []
    for feature, label in [
        ("Extracurricular", "Extracurricular"),
        ("Sports", "Sports"),
        ("Music", "Music"),
        ("Volunteering", "Volunteering"),
    ]:
        if int(record.get(feature, 0)):
            activities.append(label)

    return {
        "student_id": student_id,
        "display_name": f"Student {student_id}",
        "actual_grade": actual_grade,
        "prediction": prediction,
        "gpa": round(float(record.get("GPA", 0)), 2),
        "age": int(record.get("Age", 0)),
        "gender": GENDER_LABELS.get(int(record.get("Gender", 0)), "Unknown"),
        "parental_education": EDUCATION_LABELS[int(record.get("ParentalEducation", 0))],
        "activities": activities,
        "metrics": metrics,
    }


def build_admin_context():
    bootstrap_default_admin()
    user_model = get_user_model()
    dataset = get_dataset()
    model = get_model()

    users = [
        {
            "username": user.username,
            "role": get_user_role(user) or "lecturer",
            "role_style": ROLE_STYLE_MAP.get(get_user_role(user) or "lecturer", "bg-slate-100 text-slate-700"),
            "last_login": user.last_login.strftime("%Y-%m-%d %H:%M") if user.last_login else "Never",
        }
        for user in user_model.objects.select_related("profile").order_by("username")
    ]

    model_metrics = None
    if model is not None:
        best_result = model.comparison_results[model.best_model_name]
        model_metrics = {
            "name": model.best_model_name,
            "accuracy": f"{best_result['accuracy']:.1%}",
            "precision": f"{best_result['precision']:.1%}",
            "recall": f"{best_result['recall']:.1%}",
            "f1_score": f"{best_result['f1_score']:.1%}",
        }

    activity_logs = [
        {
            "title": "User database loaded",
            "description": f"Loaded {len(users)} Django-backed accounts from SQLite.",
            "icon": "shield_person",
        },
        {
            "title": "Dataset available",
            "description": f"Current academic dataset contains {len(dataset):,} rows.",
            "icon": "dataset",
        },
        {
            "title": "Model status",
            "description": (
                f"Best saved classifier is {model.best_model_name}."
                if model is not None
                else "No trained model was found. The UI will train one on first use."
            ),
            "icon": "model_training",
        },
    ]

    intervention_queryset = InterventionRecord.objects.select_related("user", "dataset")
    intervention_analytics = build_intervention_analytics(
        intervention_queryset,
        "All lecturer intervention records",
    )
    activity_logs.append(
        {
            "title": "Intervention tracking",
            "description": f"{intervention_analytics['total_count']:,} intervention records are currently stored across lecturer accounts.",
            "icon": "fact_check",
        }
    )

    return {
        "users": users,
        "dataset_rows": len(dataset),
        "model_metrics": model_metrics,
        "activity_logs": activity_logs,
        "total_users": len(users),
        "intervention_analytics": intervention_analytics,
        "recent_interventions": intervention_analytics["recent_records"],
    }