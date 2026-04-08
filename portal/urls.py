from django.urls import path

from . import views


app_name = "portal"

urlpatterns = [
    path("", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("dashboard/", views.dashboard_view, name="dashboard"),
    path("interventions/", views.intervention_history_view, name="intervention_history"),
    path("dataset/", views.dataset_upload_view, name="dataset_upload"),
    path("dataset/confirm/<int:pk>/", views.dataset_confirm_view, name="dataset_confirm"),
    path("predict/", views.predict_view, name="predict"),
    path("api/predict/", views.predict_api_view, name="predict_api"),
    path("api/secure/predict/", views.predict_token_api_view, name="predict_token_api"),
    path("upload/", views.batch_upload_view, name="batch_upload"),
    path("api/upload/", views.batch_upload_api_view, name="batch_upload_api"),
    path("api/secure/upload/", views.batch_upload_token_api_view, name="batch_upload_token_api"),
    path("api/token/issue/", views.issue_api_token_view, name="issue_api_token"),
    path("api/token/revoke/", views.revoke_api_token_view, name="revoke_api_token"),
    path("profile/image/", views.profile_image_upload_view, name="profile_image_upload"),
    path("students/<int:student_id>/", views.student_detail_view, name="student_detail"),
    path("admin-console/", views.admin_console_view, name="admin_console"),
]