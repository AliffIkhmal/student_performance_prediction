import csv
import json
from functools import wraps

from django.contrib import messages
from django.contrib.auth import get_user_model, login, logout
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

import pandas as pd

from .column_mapping import (
    ALL_DATASET_COLUMNS,
    DATASET_REQUIRED_COLUMNS,
    apply_column_mapping,
    resolve_columns,
)
from .auth_utils import (
    authenticate_api_token,
    bootstrap_default_admin,
    extract_api_token,
    get_api_token_summary,
    get_user_role,
    issue_api_token,
    revoke_api_token,
)
from .forms import (
    AdminUserCreateForm,
    AdminUserRemoveForm,
    BatchUploadForm,
    DatasetMappingForm,
    DatasetUploadForm,
    InterventionHistoryFilterForm,
    InterventionCreateForm,
    InterventionOutcomeForm,
    PortalLoginForm,
    ProfileImageUploadForm,
    StudentPredictionForm,
)
from .models import InterventionRecord, LecturerDataset, UserProfile
from .services import (
    StudentNotFoundError,
    build_admin_context,
    build_batch_context,
    build_dashboard_context,
    build_intervention_history_context,
    build_prediction_context,
    build_dataset_info,
    get_active_dataset_record,
    get_student_context,
    validate_dataset_frame,
)


def redirect_for_role(role):
    if role == UserProfile.ROLE_ADMIN:
        return redirect("portal:admin_console")
    return redirect("portal:dashboard")


def build_form_error_payload(form):
    return {
        field: [str(message) for message in messages]
        for field, messages in form.errors.items()
    }


def build_api_error(message, status=400, errors=None):
    return JsonResponse(
        {
            "ok": False,
            "errors": errors or {"__all__": [message]},
        },
        status=status,
    )


def build_api_access_context(request):
    snapshot = get_api_token_summary(request.user)
    snapshot.update(
        {
            "issue_url": reverse("portal:issue_api_token"),
            "revoke_url": reverse("portal:revoke_api_token"),
            "secure_predict_url": request.build_absolute_uri(reverse("portal:predict_token_api")),
            "secure_upload_url": request.build_absolute_uri(reverse("portal:batch_upload_token_api")),
        }
    )
    return snapshot


def parse_json_request(request):
    if not request.content_type.startswith("application/json"):
        return None

    try:
        return json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def session_api_role_required(*allowed_roles):
    def decorator(view_func):
        @wraps(view_func)
        def wrapped_view(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return build_api_error(
                    "Authentication required. Sign in again to continue.",
                    status=401,
                )

            role = get_user_role(request.user)
            if role not in allowed_roles:
                return build_api_error(
                    "You do not have access to that API endpoint.",
                    status=403,
                )

            return view_func(request, *args, **kwargs)

        return wrapped_view

    return decorator


def api_token_role_required(*allowed_roles):
    def decorator(view_func):
        @csrf_exempt
        @wraps(view_func)
        def wrapped_view(request, *args, **kwargs):
            raw_token = extract_api_token(request)
            if not raw_token:
                return build_api_error(
                    "Provide a Bearer token in the Authorization header or X-API-Token header.",
                    status=401,
                )

            user = authenticate_api_token(raw_token)
            if user is None:
                return build_api_error(
                    "Invalid or revoked API token.",
                    status=401,
                )

            role = get_user_role(user)
            if role not in allowed_roles:
                return build_api_error(
                    "You do not have access to that API endpoint.",
                    status=403,
                )

            request.api_user = user
            return view_func(request, *args, **kwargs)

        return wrapped_view

    return decorator


def role_required(*allowed_roles):
    def decorator(view_func):
        @login_required(login_url="portal:login")
        @wraps(view_func)
        def wrapped_view(request, *args, **kwargs):
            role = get_user_role(request.user)
            if role not in allowed_roles:
                messages.error(request, "You do not have access to that page.")
                return redirect("portal:dashboard")

            return view_func(request, *args, **kwargs)

        return wrapped_view

    return decorator


def login_view(request):
    bootstrap_default_admin()

    if request.user.is_authenticated:
        return redirect_for_role(get_user_role(request.user))

    form = PortalLoginForm(request=request, data=request.POST or None)
    next_url = request.POST.get("next") or request.GET.get("next") or ""

    if request.method == "POST" and form.is_valid():
        user = form.get_user()
        login(request, user)
        if form.cleaned_data.get("keep_session_active"):
            request.session.set_expiry(60 * 60 * 8)
        else:
            request.session.set_expiry(0)
        role = get_user_role(user)
        messages.success(request, f"Signed in as {role}.")
        if next_url and url_has_allowed_host_and_scheme(
            next_url,
            allowed_hosts={request.get_host()},
            require_https=request.is_secure(),
        ):
            return redirect(next_url)
        return redirect_for_role(role)

    return render(
        request,
        "portal/login.html",
        {
            "hide_chrome": True,
            "page_title": "Scholar Bento | Secure Login",
            "login_form": form,
            "next_url": next_url,
        },
    )


def logout_view(request):
    logout(request)
    messages.success(request, "You have been signed out.")
    return redirect("portal:login")


@require_POST
@login_required(login_url="portal:login")
def profile_image_upload_view(request):
    form = ProfileImageUploadForm(request.POST, request.FILES)
    redirect_to = request.POST.get("next") or request.META.get("HTTP_REFERER") or reverse("portal:dashboard")

    if form.is_valid():
        profile = request.user.profile
        profile.profile_image = form.cleaned_data["profile_image"]
        profile.save(update_fields=["profile_image"])
        messages.success(request, "Profile image updated successfully.")
    else:
        first_error = form.errors.get("profile_image") or form.non_field_errors()
        error_text = first_error[0] if first_error else "Profile image could not be updated."
        messages.error(request, error_text)

    if url_has_allowed_host_and_scheme(
        redirect_to,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        return redirect(redirect_to)
    return redirect("portal:dashboard")


@role_required("lecturer", "admin")
def dashboard_view(request):
    context = build_dashboard_context(request.GET, user=request.user)
    context.update(
        {
            "page_title": "Lecturer Dashboard",
            "page_label": "Overview",
            "page_heading": "Student Performance Dashboard",
            "page_description": "A Django bento dashboard for early-risk monitoring, grade distribution, model insights, and academic-period comparisons.",
            "api_access": build_api_access_context(request),
        }
    )
    return render(request, "portal/dashboard.html", context)


@role_required("lecturer", "admin")
def intervention_history_view(request):
    context = build_intervention_history_context(request.user, request.GET)

    if request.GET.get("export") == "csv":
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="intervention-history.csv"'
        writer = csv.writer(response)
        writer.writerow([
            "Staff",
            "Student ID",
            "Title",
            "Category",
            "Severity",
            "Status",
            "Outcome",
            "Target Feature",
            "Feature Value",
            "SHAP Priority",
            "Predicted Grade",
            "Predicted Risk",
            "Predicted Risk Score",
            "Review Date",
            "Plan Note",
            "Outcome Note",
            "Created At",
            "Updated At",
        ])
        for record in context["records"]:
            writer.writerow([
                record.user.username,
                record.student_id,
                record.title,
                record.get_category_display(),
                record.get_severity_display(),
                record.get_status_display(),
                record.get_outcome_display(),
                record.target_feature_label,
                record.feature_value if record.feature_value is not None else "",
                record.impact_share,
                record.predicted_grade,
                record.predicted_risk_label,
                record.predicted_risk_score,
                record.review_date.isoformat() if record.review_date else "",
                record.note,
                record.outcome_note,
                record.created_at.strftime("%Y-%m-%d %H:%M"),
                record.updated_at.strftime("%Y-%m-%d %H:%M"),
            ])
        return response

    context.update(
        {
            "page_title": "Intervention History",
            "page_label": "Interventions",
            "page_heading": "Intervention History",
            "page_description": "Filter, review, and export tracked intervention outcomes over time.",
        }
    )
    return render(request, "portal/intervention_history.html", context)


@role_required("lecturer", "admin")
def predict_view(request):
    form = StudentPredictionForm(request.POST or None, user=request.user)
    prediction = None

    if request.method == "POST":
        if form.is_valid():
            prediction = build_prediction_context(form.to_feature_payload())
            messages.success(request, "Prediction generated successfully.")
        else:
            messages.error(request, "Fix the highlighted prediction fields and try again.")

    return render(
        request,
        "portal/predict.html",
        {
            "page_title": "Single Student Prediction",
            "page_label": "Predict",
            "page_heading": "Single Student Prediction",
            "page_description": "Use the current feature set from your ML model and generate a grade prediction with explanations.",
            "prediction_form": form,
            "prediction": prediction,
        },
    )


@role_required("lecturer", "admin")
def batch_upload_view(request):
    form = BatchUploadForm(request.POST or None, request.FILES or None)
    batch_result = None
    active_dataset = get_active_dataset_record(request.user)
    preferred_mapping = active_dataset.column_mapping if active_dataset is not None else None

    if request.method == "POST":
        if form.is_valid():
            batch_result = build_batch_context(
                form.cleaned_data["dataset"],
                preferred_mapping=preferred_mapping,
            )
            if batch_result.get("error"):
                messages.error(request, batch_result["error"])
            else:
                messages.success(
                    request,
                    f"Processed {batch_result['processed_count']} rows from {batch_result['file_name']}.",
                )
        else:
            messages.error(request, "Fix the upload error and submit the file again.")

    return render(
        request,
        "portal/batch_upload.html",
        {
            "page_title": "Batch Prediction",
            "page_label": "Upload",
            "page_heading": "Batch Prediction Engine",
            "page_description": "Validate a CSV upload, run predictions in bulk, and review the highest-risk records first.",
            "batch_upload_form": form,
            "active_dataset": build_dataset_info(active_dataset),
            "batch_result": batch_result,
        },
    )


@role_required("lecturer", "admin")
def dataset_upload_view(request):
    form = DatasetUploadForm(request.POST or None, request.FILES or None)
    active_dataset = get_active_dataset_record(request.user)

    if request.method == "POST":
        if form.is_valid():
            uploaded_file = form.cleaned_data["dataset"]
            uploaded_file.seek(0)
            try:
                dataset_preview = pd.read_csv(uploaded_file)
            except Exception as exc:
                form.add_error("dataset", f"Could not read the uploaded CSV file: {exc}")
            else:
                mapping_result = resolve_columns(
                    dataset_preview,
                    canonical_columns=ALL_DATASET_COLUMNS,
                )
                uploaded_file.seek(0)
                lecturer_dataset = LecturerDataset.objects.create(
                    user=request.user,
                    file=uploaded_file,
                    original_filename=uploaded_file.name,
                    column_mapping=mapping_result["auto_mapped"],
                    row_count=len(dataset_preview),
                    is_active=False,
                )
                messages.success(
                    request,
                    "Dataset uploaded. Review the detected column mapping before activation.",
                )
                return redirect("portal:dataset_confirm", pk=lecturer_dataset.pk)
        else:
            messages.error(request, "Fix the upload error and submit the dataset again.")

    return render(
        request,
        "portal/dataset_upload.html",
        {
            "page_title": "My Dataset",
            "page_label": "Dataset",
            "page_heading": "My Dataset",
            "page_description": "Upload your own lecturer dataset, review the detected column mapping, and activate it for your private dashboard.",
            "dataset_upload_form": form,
            "active_dataset": build_dataset_info(active_dataset),
        },
    )


@role_required("lecturer", "admin")
def dataset_confirm_view(request, pk):
    lecturer_dataset = get_object_or_404(LecturerDataset, pk=pk, user=request.user)

    try:
        raw_dataset = lecturer_dataset.load_dataframe(apply_mapping=False)
    except Exception as exc:
        messages.error(request, f"Could not open the uploaded dataset: {exc}")
        return redirect("portal:dataset_upload")

    selected_mapping = lecturer_dataset.column_mapping or resolve_columns(
        raw_dataset,
        canonical_columns=ALL_DATASET_COLUMNS,
    )["auto_mapped"]

    form = DatasetMappingForm(
        request.POST or None,
        csv_columns=list(raw_dataset.columns),
        selected_mapping=selected_mapping,
        canonical_columns=ALL_DATASET_COLUMNS,
        required_columns=DATASET_REQUIRED_COLUMNS,
    )

    if request.method == "POST":
        if form.is_valid():
            column_mapping = form.cleaned_data["column_mapping"]
            try:
                mapped_dataset = lecturer_dataset.load_dataframe(apply_mapping=False)
                mapped_dataset = apply_column_mapping(mapped_dataset, column_mapping)
            except ValueError as exc:
                form.add_error(None, str(exc))
            else:
                normalized_dataset, errors = validate_dataset_frame(
                    mapped_dataset,
                    DATASET_REQUIRED_COLUMNS,
                )
                if errors:
                    form.add_error(None, " ".join(errors))
                else:
                    lecturer_dataset.column_mapping = column_mapping
                    lecturer_dataset.row_count = len(normalized_dataset)
                    lecturer_dataset.activate()
                    messages.success(
                        request,
                        f"{lecturer_dataset.original_filename} is now your active dataset.",
                    )
                    return redirect("portal:dashboard")
        else:
            messages.error(request, "Review the column mapping and fix the highlighted fields.")

    for row in form.mapping_rows:
        row["bound_field"] = form[row["field_name"]]

    return render(
        request,
        "portal/dataset_confirm.html",
        {
            "page_title": "Confirm Dataset Mapping",
            "page_label": "Dataset",
            "page_heading": "Confirm Dataset Mapping",
            "page_description": "Check each detected column before activating this dataset for your dashboard and student detail views.",
            "mapping_form": form,
            "mapping_rows": form.mapping_rows,
            "lecturer_dataset": lecturer_dataset,
            "csv_columns": list(raw_dataset.columns),
            "unmatched_columns": [
                column_name
                for column_name in raw_dataset.columns
                if column_name not in form.cleaned_data.get("column_mapping", selected_mapping).values()
            ] if request.method == "POST" else [
                column_name
                for column_name in raw_dataset.columns
                if column_name not in selected_mapping.values()
            ],
        },
    )


@require_POST
@session_api_role_required("lecturer", "admin")
def predict_api_view(request):
    payload = parse_json_request(request) or request.POST
    form = StudentPredictionForm(payload, user=request.user)

    if not form.is_valid():
        return build_api_error(
            "Prediction request contains invalid fields.",
            status=400,
            errors=build_form_error_payload(form),
        )

    prediction = build_prediction_context(form.to_feature_payload())
    return JsonResponse({"ok": True, "result": prediction})


@require_POST
@session_api_role_required("lecturer", "admin")
def batch_upload_api_view(request):
    form = BatchUploadForm(request.POST, request.FILES)

    if not form.is_valid():
        return build_api_error(
            "Batch upload request contains invalid fields.",
            status=400,
            errors=build_form_error_payload(form),
        )

    active_dataset = get_active_dataset_record(request.user)
    batch_result = build_batch_context(
        form.cleaned_data["dataset"],
        preferred_mapping=active_dataset.column_mapping if active_dataset is not None else None,
    )
    if batch_result.get("error"):
        return build_api_error(
            batch_result["error"],
            status=400,
            errors={"dataset": [batch_result["error"]]},
        )

    return JsonResponse({"ok": True, "result": batch_result})


@api_token_role_required("lecturer", "admin")
@require_POST
def predict_token_api_view(request):
    payload = parse_json_request(request) or request.POST
    form = StudentPredictionForm(payload, user=request.api_user)

    if not form.is_valid():
        return build_api_error(
            "Prediction request contains invalid fields.",
            status=400,
            errors=build_form_error_payload(form),
        )

    prediction = build_prediction_context(form.to_feature_payload())
    return JsonResponse({"ok": True, "result": prediction})


@api_token_role_required("lecturer", "admin")
@require_POST
def batch_upload_token_api_view(request):
    form = BatchUploadForm(request.POST, request.FILES)

    if not form.is_valid():
        return build_api_error(
            "Batch upload request contains invalid fields.",
            status=400,
            errors=build_form_error_payload(form),
        )

    active_dataset = get_active_dataset_record(request.api_user)
    batch_result = build_batch_context(
        form.cleaned_data["dataset"],
        preferred_mapping=active_dataset.column_mapping if active_dataset is not None else None,
    )
    if batch_result.get("error"):
        return build_api_error(
            batch_result["error"],
            status=400,
            errors={"dataset": [batch_result["error"]]},
        )

    return JsonResponse({"ok": True, "result": batch_result})


@require_POST
@session_api_role_required("lecturer", "admin")
def issue_api_token_view(request):
    token, raw_token = issue_api_token(request.user)
    return JsonResponse(
        {
            "ok": True,
            "result": {
                "token": raw_token,
                "token_prefix": token.token_prefix,
                "created_at": token.created_at.strftime("%Y-%m-%d %H:%M"),
                "last_used_at": "Never",
                "authorization_header": f"Bearer {raw_token}",
            },
        }
    )


@require_POST
@session_api_role_required("lecturer", "admin")
def revoke_api_token_view(request):
    if not revoke_api_token(request.user):
        return build_api_error(
            "No active API token was found for this account.",
            status=404,
        )

    return JsonResponse({"ok": True, "result": {"revoked": True}})


@role_required("lecturer", "admin")
def student_detail_view(request, student_id):
    active_dataset = get_active_dataset_record(request.user)

    try:
        student = get_student_context(student_id, user=request.user)
    except StudentNotFoundError as exc:
        messages.error(request, str(exc))
        return redirect("portal:dashboard")

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "track_intervention":
            track_form = InterventionCreateForm(request.POST)
            if track_form.is_valid():
                cleaned_data = track_form.cleaned_data
                duplicate = InterventionRecord.objects.filter(
                    user=request.user,
                    student_id=student_id,
                    title=cleaned_data["title"],
                    status__in=[
                        InterventionRecord.Status.PLANNED,
                        InterventionRecord.Status.IN_PROGRESS,
                    ],
                ).first()
                if duplicate is not None:
                    messages.info(request, f"{duplicate.title} is already being tracked for this student.")
                else:
                    InterventionRecord.objects.create(
                        user=request.user,
                        dataset=active_dataset,
                        student_id=student_id,
                        title=cleaned_data["title"],
                        category=cleaned_data["category"],
                        severity=cleaned_data["severity"],
                        target_feature=cleaned_data.get("target_feature", ""),
                        feature_value=cleaned_data.get("feature_value"),
                        shap_value=cleaned_data.get("shap_value"),
                        impact_share=cleaned_data.get("impact_share") or 0,
                        predicted_grade=cleaned_data["predicted_grade"],
                        predicted_risk_label=cleaned_data["predicted_risk_label"],
                        predicted_risk_score=cleaned_data["predicted_risk_score"],
                        note=cleaned_data.get("note", ""),
                        review_date=cleaned_data.get("review_date"),
                    )
                    messages.success(request, f"Tracked intervention: {cleaned_data['title']}.")
                return redirect("portal:student_detail", student_id=student_id)

            messages.error(request, "The selected intervention could not be tracked. Try again.")

        if action == "update_intervention":
            record = get_object_or_404(
                InterventionRecord,
                pk=request.POST.get("record_id"),
                user=request.user,
                student_id=student_id,
            )
            outcome_form = InterventionOutcomeForm(request.POST)
            if outcome_form.is_valid():
                cleaned_data = outcome_form.cleaned_data
                record.status = cleaned_data["status"]
                record.outcome = cleaned_data["outcome"]
                record.outcome_note = cleaned_data.get("outcome_note", "")
                record.review_date = cleaned_data.get("review_date")
                record.save(update_fields=["status", "outcome", "outcome_note", "review_date", "updated_at"])
                messages.success(request, f"Updated intervention outcome for {record.title}.")
                return redirect("portal:student_detail", student_id=student_id)

            messages.error(request, "The intervention outcome could not be updated. Try again.")

    tracked_interventions = list(
        InterventionRecord.objects.filter(user=request.user, student_id=student_id).select_related("dataset")
    )
    for recommendation in student["prediction"]["recommendations"]:
        recommendation["track_form"] = InterventionCreateForm(
            initial={
                "title": recommendation["title"],
                "category": recommendation["category"],
                "severity": recommendation["severity"],
                "target_feature": recommendation.get("target_feature", ""),
                "feature_value": recommendation.get("feature_value"),
                "shap_value": recommendation.get("shap_value"),
                "impact_share": recommendation.get("impact_share", 0),
                "predicted_grade": student["prediction"]["predicted_grade"],
                "predicted_risk_label": student["prediction"]["risk_label"],
                "predicted_risk_score": student["prediction"]["risk_score"],
            }
        )

    for record in tracked_interventions:
        record.outcome_form = InterventionOutcomeForm(
            initial={
                "status": record.status,
                "outcome": record.outcome,
                "outcome_note": record.outcome_note,
                "review_date": record.review_date,
            }
        )

    return render(
        request,
        "portal/student_detail.html",
        {
            "page_title": f"Student {student_id} Detail",
            "page_label": "Student Detail",
            "page_heading": student["display_name"],
            "page_description": "Current academic signals, SHAP-ranked intervention targets, and tracked intervention outcomes.",
            "student": student,
            "tracked_interventions": tracked_interventions,
        },
    )


@role_required("admin",)
def admin_console_view(request):
    add_user_form = AdminUserCreateForm()

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "add_user":
            add_user_form = AdminUserCreateForm(request.POST)
            if add_user_form.is_valid():
                user = add_user_form.save()
                messages.success(request, f"User {user.username} created successfully.")
                return redirect("portal:admin_console")
            messages.error(request, "Fix the user creation form and try again.")

        if action == "remove_user":
            remove_user_form = AdminUserRemoveForm(request.POST, current_user=request.user)
            if remove_user_form.is_valid():
                username = remove_user_form.cleaned_data["username"]
                remove_user_form.save()
                messages.success(request, f"Removed user {username}.")
            else:
                first_error = remove_user_form.non_field_errors() or remove_user_form.errors.get("username")
                error_text = first_error[0] if first_error else "Could not remove the selected user."
                messages.error(request, error_text)
            return redirect("portal:admin_console")

    context = build_admin_context()
    context.update(
        {
            "page_title": "Admin Console",
            "page_label": "Admin",
            "page_heading": "Scholar Bento Admin",
            "page_description": "Manage lecturer access, inspect model status, and review the dataset snapshot used by the Django prototype.",
            "add_user_form": add_user_form,
        }
    )
    return render(request, "portal/admin_console.html", context)