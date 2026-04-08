import io
import json
import shutil
import tempfile

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse
from PIL import Image

import pandas as pd

from .auth_utils import get_user_role, set_user_role
from .column_mapping import DATASET_REQUIRED_COLUMNS, resolve_columns
from .models import InterventionRecord, LecturerDataset, UserProfile


class ColumnMappingTests(TestCase):
    def test_resolve_columns_detects_standard_aliases(self):
        dataset = pd.DataFrame(
            columns=[
                "student_id",
                "student_age",
                "sex",
                "parent_education",
                "study_hours",
                "absence_count",
                "parent_support",
                "clubs",
                "athletics",
                "music",
                "volunteerwork",
                "cgpa",
                "final_grade",
            ]
        )

        result = resolve_columns(dataset, canonical_columns=DATASET_REQUIRED_COLUMNS)

        self.assertEqual(result["auto_mapped"]["StudentID"], "student_id")
        self.assertEqual(result["auto_mapped"]["Age"], "student_age")
        self.assertEqual(result["auto_mapped"]["StudyTimeWeekly"], "study_hours")
        self.assertEqual(result["auto_mapped"]["GPA"], "cgpa")
        self.assertEqual(result["auto_mapped"]["GradeClass"], "final_grade")

    def test_resolve_columns_reports_unmapped_fields(self):
        dataset = pd.DataFrame(columns=["foo", "bar"])

        result = resolve_columns(dataset, canonical_columns=["Age", "GPA"])

        self.assertEqual(result["unmapped_canonical"], ["Age", "GPA"])
        self.assertEqual(result["unmapped_csv"], ["foo", "bar"])


class PortalSmokeTests(TestCase):
    def setUp(self):
        self.media_dir = tempfile.mkdtemp()
        self.override = self.settings(MEDIA_ROOT=self.media_dir)
        self.override.enable()

        user_model = get_user_model()
        self.lecturer = user_model.objects.create_user(
            username="lecturer-demo",
            password="password123",
        )
        set_user_role(self.lecturer, UserProfile.ROLE_LECTURER)

        self.admin = user_model.objects.create_user(
            username="admin-demo",
            password="password123",
            is_staff=True,
        )
        set_user_role(self.admin, UserProfile.ROLE_ADMIN)

    def tearDown(self):
        self.override.disable()
        shutil.rmtree(self.media_dir, ignore_errors=True)

    def canonical_dataset_csv(self, include_session=False):
        headers = [
            "StudentID",
            "Age",
            "Gender",
            "ParentalEducation",
            "StudyTimeWeekly",
            "Absences",
            "ParentalSupport",
            "Extracurricular",
            "Sports",
            "Music",
            "Volunteering",
            "GPA",
            "GradeClass",
        ]
        if include_session:
            headers.append("AcademicSession")

        rows = [
            [101, 17, 0, 2, 11.5, 3, 3, 1, 0, 1, 0, 3.2, 1],
            [102, 18, 1, 1, 6.0, 11, 1, 0, 0, 0, 0, 2.1, 3],
        ]
        if include_session:
            rows[0].append("2024/2025")
            rows[1].append("2023/2024")

        frame = pd.DataFrame(rows, columns=headers)
        return frame.to_csv(index=False)

    def alias_dataset_csv(self, custom_study_header=False, include_session=False):
        study_header = "focus_hours" if custom_study_header else "study_hours"
        headers = [
            "student_id",
            "student_age",
            "sex",
            "parent_education",
            study_header,
            "absence_count",
            "parent_support",
            "clubs",
            "athletics",
            "music",
            "volunteerwork",
            "cgpa",
            "final_grade",
        ]
        if include_session:
            headers.append("session")

        rows = [
            [201, 17, 0, 2, 12.0, 2, 3, 1, 0, 1, 0, 3.5, 1],
            [202, 18, 1, 1, 5.5, 12, 1, 0, 0, 0, 1, 2.0, 4],
        ]
        if include_session:
            rows[0].append("2024/2025")
            rows[1].append("2023/2024")

        frame = pd.DataFrame(rows, columns=headers)
        return frame.to_csv(index=False)

    def batch_alias_csv(self, custom_study_header=False):
        study_header = "focus_hours" if custom_study_header else "study_hours"
        headers = [
            "student_id",
            "student_age",
            "sex",
            "parent_education",
            study_header,
            "absence_count",
            "parent_support",
            "clubs",
            "athletics",
            "music",
            "volunteerwork",
        ]
        rows = [
            [301, 17, 0, 2, 10.5, 4, 3, 1, 0, 1, 0],
        ]
        return pd.DataFrame(rows, columns=headers).to_csv(index=False)

    def upload_dataset(self, csv_content, filename="lecturer-dataset.csv"):
        upload = SimpleUploadedFile(
            filename,
            csv_content.encode("utf-8"),
            content_type="text/csv",
        )
        response = self.client.post(reverse("portal:dataset_upload"), {"dataset": upload})
        dataset = LecturerDataset.objects.latest("id")
        return response, dataset

    def confirm_dataset(self, dataset, column_mapping):
        post_data = {
            f"map_{canonical_name}": actual_name
            for canonical_name, actual_name in column_mapping.items()
        }
        return self.client.post(
            reverse("portal:dataset_confirm", args=[dataset.pk]),
            post_data,
        )

    def activate_dataset(self, csv_content, mapping=None, filename="active.csv"):
        upload = SimpleUploadedFile(
            filename,
            csv_content.encode("utf-8"),
            content_type="text/csv",
        )
        dataset = LecturerDataset.objects.create(
            user=self.lecturer,
            file=upload,
            original_filename=filename,
            column_mapping=mapping or {},
            row_count=2,
            is_active=False,
        )
        dataset.activate()
        return dataset

    def image_upload(self, filename="avatar.png"):
        image_bytes = io.BytesIO()
        image = Image.new("RGB", (1, 1), color=(5, 17, 37))
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)
        return SimpleUploadedFile(filename, image_bytes.getvalue(), content_type="image/png")

    def test_login_page_loads(self):
        response = self.client.get(reverse("portal:login"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Scholar Bento")

    def test_dashboard_requires_session_auth(self):
        response = self.client.get(reverse("portal:dashboard"))
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(
            response,
            f"{reverse('portal:login')}?next={reverse('portal:dashboard')}",
        )

    def test_logged_in_lecturer_can_open_dashboard(self):
        self.client.force_login(self.lecturer)

        response = self.client.get(reverse("portal:dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Student Performance Dashboard")

    def test_database_login_redirects_to_dashboard(self):
        response = self.client.post(
            reverse("portal:login"),
            {
                "username": "lecturer-demo",
                "password": "password123",
                "keep_session_active": "on",
            },
        )
        self.assertRedirects(response, reverse("portal:dashboard"))

    def test_login_page_contains_password_toggle(self):
        response = self.client.get(reverse("portal:login"))
        self.assertContains(response, "data-password-toggle")

    def test_dashboard_prompts_for_personal_dataset_when_none_uploaded(self):
        self.client.force_login(self.lecturer)

        response = self.client.get(reverse("portal:dashboard"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No personal dataset yet")
        self.assertContains(response, "Open My Dataset")

    def test_dataset_upload_redirects_to_mapping_confirmation(self):
        self.client.force_login(self.lecturer)

        response, dataset = self.upload_dataset(
            self.alias_dataset_csv(),
            filename="alias-dataset.csv",
        )

        self.assertRedirects(
            response,
            reverse("portal:dataset_confirm", args=[dataset.pk]),
        )
        self.assertFalse(dataset.is_active)
        self.assertEqual(dataset.column_mapping["Age"], "student_age")
        self.assertEqual(dataset.column_mapping["GPA"], "cgpa")

    def test_dataset_confirm_activates_and_dashboard_uses_uploaded_dataset(self):
        self.client.force_login(self.lecturer)
        response, dataset = self.upload_dataset(
            self.alias_dataset_csv(custom_study_header=True),
            filename="custom-dataset.csv",
        )
        self.assertEqual(response.status_code, 302)

        confirm_response = self.confirm_dataset(
            dataset,
            {
                "StudentID": "student_id",
                "Age": "student_age",
                "Gender": "sex",
                "ParentalEducation": "parent_education",
                "StudyTimeWeekly": "focus_hours",
                "Absences": "absence_count",
                "ParentalSupport": "parent_support",
                "Extracurricular": "clubs",
                "Sports": "athletics",
                "Music": "music",
                "Volunteering": "volunteerwork",
                "GPA": "cgpa",
                "GradeClass": "final_grade",
            },
        )

        self.assertRedirects(confirm_response, reverse("portal:dashboard"))
        dataset.refresh_from_db()
        self.assertTrue(dataset.is_active)

        dashboard_response = self.client.get(reverse("portal:dashboard"))
        self.assertContains(dashboard_response, "custom-dataset.csv")
        self.assertContains(dashboard_response, "Student 202")

    def test_dashboard_shows_period_filter_hint_without_temporal_columns(self):
        self.activate_dataset(self.canonical_dataset_csv(), filename="plain.csv")
        self.client.force_login(self.lecturer)

        response = self.client.get(reverse("portal:dashboard"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Period filters are ready, but this dataset does not include time fields yet.")

    def test_dashboard_can_compare_academic_sessions(self):
        self.activate_dataset(
            self.canonical_dataset_csv(include_session=True),
            filename="sessions.csv",
        )
        self.client.force_login(self.lecturer)

        response = self.client.get(
            reverse("portal:dashboard"),
            {
                "period_dimension": "AcademicSession",
                "period_value": "2024/2025",
                "compare_value": "2023/2024",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Academic Session: 2024/2025")
        self.assertContains(response, "Compare the current view against 2023/2024.")

    def test_dashboard_shows_intervention_analytics(self):
        self.activate_dataset(self.canonical_dataset_csv(), filename="analytics.csv")
        InterventionRecord.objects.create(
            user=self.lecturer,
            student_id=102,
            title="Attendance intervention",
            category=InterventionRecord.Category.ATTENDANCE,
            severity=InterventionRecord.Severity.URGENT,
            status=InterventionRecord.Status.COMPLETED,
            outcome=InterventionRecord.Outcome.IMPROVED,
            review_date="2026-04-10",
        )
        self.client.force_login(self.lecturer)

        response = self.client.get(reverse("portal:dashboard"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Intervention Outcome Analytics")
        self.assertContains(response, "Improvement Rate")

    def test_intervention_history_page_filters_records(self):
        InterventionRecord.objects.create(
            user=self.lecturer,
            student_id=101,
            title="Attendance intervention",
            category=InterventionRecord.Category.ATTENDANCE,
            severity=InterventionRecord.Severity.URGENT,
            status=InterventionRecord.Status.COMPLETED,
            outcome=InterventionRecord.Outcome.IMPROVED,
        )
        InterventionRecord.objects.create(
            user=self.lecturer,
            student_id=102,
            title="Study plan reset",
            category=InterventionRecord.Category.STUDY,
            severity=InterventionRecord.Severity.RECOMMENDED,
            status=InterventionRecord.Status.IN_PROGRESS,
            outcome=InterventionRecord.Outcome.PENDING,
        )
        self.client.force_login(self.lecturer)

        response = self.client.get(reverse("portal:intervention_history"), {"outcome": InterventionRecord.Outcome.IMPROVED})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Attendance intervention")
        self.assertNotContains(response, "Study plan reset")

    def test_intervention_history_export_returns_csv(self):
        InterventionRecord.objects.create(
            user=self.lecturer,
            student_id=101,
            title="Attendance intervention",
            category=InterventionRecord.Category.ATTENDANCE,
            severity=InterventionRecord.Severity.URGENT,
            status=InterventionRecord.Status.COMPLETED,
            outcome=InterventionRecord.Outcome.IMPROVED,
        )
        self.client.force_login(self.lecturer)

        response = self.client.get(reverse("portal:intervention_history"), {"export": "csv"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv")
        self.assertIn("Attendance intervention", response.content.decode("utf-8"))

    def test_profile_image_upload_updates_profile(self):
        self.client.force_login(self.lecturer)

        response = self.client.post(
            reverse("portal:profile_image_upload"),
            {
                "next": reverse("portal:dashboard"),
                "profile_image": self.image_upload(),
            },
        )

        self.assertRedirects(response, reverse("portal:dashboard"))
        self.lecturer.refresh_from_db()
        self.assertTrue(bool(self.lecturer.profile.profile_image))

    def test_student_detail_reads_from_active_dataset(self):
        self.activate_dataset(self.canonical_dataset_csv(), filename="detail.csv")
        self.client.force_login(self.lecturer)

        response = self.client.get(reverse("portal:student_detail", args=[101]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Student 101")
        self.assertContains(response, "Current Outlook")
        self.assertContains(response, "Intervention Tracker")

    def test_prediction_api_returns_targeted_intervention_metadata(self):
        self.client.force_login(self.lecturer)

        response = self.client.post(
            reverse("portal:predict_api"),
            {
                "Age": 18,
                "Gender": 1,
                "ParentalEducation": 1,
                "StudyTimeWeekly": 5.5,
                "Absences": 11,
                "ParentalSupport": 1,
                "Extracurricular": 0,
                "Sports": 0,
                "Music": 0,
                "Volunteering": 0,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertIn("target_feature", payload["result"]["recommendations"][0])
        self.assertIn("category", payload["result"]["recommendations"][0])
        self.assertIn("severity", payload["result"]["recommendations"][0])

    def test_student_detail_can_track_targeted_intervention(self):
        self.activate_dataset(self.canonical_dataset_csv(), filename="track.csv")
        self.client.force_login(self.lecturer)

        detail_response = self.client.get(reverse("portal:student_detail", args=[102]))
        recommendation = detail_response.context["student"]["prediction"]["recommendations"][0]
        prediction = detail_response.context["student"]["prediction"]

        response = self.client.post(
            reverse("portal:student_detail", args=[102]),
            {
                "action": "track_intervention",
                "title": recommendation["title"],
                "category": recommendation["category"],
                "severity": recommendation["severity"],
                "target_feature": recommendation.get("target_feature", ""),
                "feature_value": recommendation.get("feature_value") or "",
                "shap_value": recommendation.get("shap_value") or "",
                "impact_share": recommendation.get("impact_share", 0),
                "predicted_grade": prediction["predicted_grade"],
                "predicted_risk_label": prediction["risk_label"],
                "predicted_risk_score": prediction["risk_score"],
                "note": "Initial outreach booked",
                "review_date": "2026-04-15",
            },
        )

        self.assertRedirects(response, reverse("portal:student_detail", args=[102]))
        record = InterventionRecord.objects.get(user=self.lecturer, student_id=102)
        self.assertEqual(record.title, recommendation["title"])
        self.assertEqual(record.outcome, InterventionRecord.Outcome.PENDING)
        self.assertEqual(record.note, "Initial outreach booked")

    def test_student_detail_can_update_intervention_outcome(self):
        dataset = self.activate_dataset(self.canonical_dataset_csv(), filename="outcome.csv")
        record = InterventionRecord.objects.create(
            user=self.lecturer,
            dataset=dataset,
            student_id=102,
            title="Attendance intervention",
            category=InterventionRecord.Category.ATTENDANCE,
            severity=InterventionRecord.Severity.URGENT,
            target_feature="Absences",
            feature_value=11,
            shap_value=0.214,
            impact_share=34,
            predicted_grade="D",
            predicted_risk_label="High Risk",
            predicted_risk_score=82,
        )
        self.client.force_login(self.lecturer)

        response = self.client.post(
            reverse("portal:student_detail", args=[102]),
            {
                "action": "update_intervention",
                "record_id": record.pk,
                "status": InterventionRecord.Status.COMPLETED,
                "outcome": InterventionRecord.Outcome.IMPROVED,
                "outcome_note": "Attendance improved after the meeting.",
                "review_date": "2026-04-22",
            },
        )

        self.assertRedirects(response, reverse("portal:student_detail", args=[102]))
        record.refresh_from_db()
        self.assertEqual(record.status, InterventionRecord.Status.COMPLETED)
        self.assertEqual(record.outcome, InterventionRecord.Outcome.IMPROVED)
        self.assertEqual(record.outcome_note, "Attendance improved after the meeting.")

    def test_admin_console_blocks_lecturer_role(self):
        self.client.force_login(self.lecturer)

        response = self.client.get(reverse("portal:admin_console"))
        self.assertEqual(response.status_code, 302)

    def test_admin_console_loads_for_admin(self):
        self.client.force_login(self.admin)

        response = self.client.get(reverse("portal:admin_console"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Scholar Bento Admin")
        self.assertContains(response, "Intervention Analytics")
        self.assertContains(response, "data-password-toggle")

    def test_prediction_api_returns_json_for_valid_input(self):
        self.client.force_login(self.lecturer)

        response = self.client.post(
            reverse("portal:predict_api"),
            {
                "Age": 17,
                "Gender": 0,
                "ParentalEducation": 2,
                "StudyTimeWeekly": 10,
                "Absences": 4,
                "ParentalSupport": 3,
                "Extracurricular": 1,
                "Sports": 0,
                "Music": 1,
                "Volunteering": 0,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertIn("predicted_grade", payload["result"])
        self.assertIn("risk_score", payload["result"])

    def test_issue_api_token_returns_plaintext_once(self):
        self.client.force_login(self.lecturer)

        response = self.client.post(reverse("portal:issue_api_token"))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertIn("token", payload["result"])
        self.assertTrue(payload["result"]["token"].startswith("spp_"))

    def test_secure_prediction_api_accepts_bearer_token(self):
        self.client.force_login(self.lecturer)
        issue_response = self.client.post(reverse("portal:issue_api_token"))
        token = issue_response.json()["result"]["token"]
        self.client.logout()

        response = self.client.post(
            reverse("portal:predict_token_api"),
            data=json.dumps(
                {
                    "Age": 17,
                    "Gender": 0,
                    "ParentalEducation": 2,
                    "StudyTimeWeekly": 10,
                    "Absences": 4,
                    "ParentalSupport": 3,
                    "Extracurricular": 1,
                    "Sports": 0,
                    "Music": 1,
                    "Volunteering": 0,
                }
            ),
            content_type="application/json",
            HTTP_AUTHORIZATION=f"Bearer {token}",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertIn("predicted_grade", payload["result"])

    def test_secure_prediction_api_rejects_invalid_token(self):
        response = self.client.post(
            reverse("portal:predict_token_api"),
            data=json.dumps(
                {
                    "Age": 17,
                    "Gender": 0,
                    "ParentalEducation": 2,
                    "StudyTimeWeekly": 10,
                    "Absences": 4,
                    "ParentalSupport": 3,
                    "Extracurricular": 1,
                    "Sports": 0,
                    "Music": 1,
                    "Volunteering": 0,
                }
            ),
            content_type="application/json",
            HTTP_AUTHORIZATION="Bearer invalid-token",
        )

        self.assertEqual(response.status_code, 401)
        payload = response.json()
        self.assertFalse(payload["ok"])

    def test_revoked_token_cannot_access_secure_prediction_api(self):
        self.client.force_login(self.lecturer)
        issue_response = self.client.post(reverse("portal:issue_api_token"))
        token = issue_response.json()["result"]["token"]
        revoke_response = self.client.post(reverse("portal:revoke_api_token"))
        self.assertEqual(revoke_response.status_code, 200)
        self.client.logout()

        response = self.client.post(
            reverse("portal:predict_token_api"),
            data=json.dumps(
                {
                    "Age": 17,
                    "Gender": 0,
                    "ParentalEducation": 2,
                    "StudyTimeWeekly": 10,
                    "Absences": 4,
                    "ParentalSupport": 3,
                    "Extracurricular": 1,
                    "Sports": 0,
                    "Music": 1,
                    "Volunteering": 0,
                }
            ),
            content_type="application/json",
            HTTP_AUTHORIZATION=f"Bearer {token}",
        )

        self.assertEqual(response.status_code, 401)
        payload = response.json()
        self.assertFalse(payload["ok"])

    def test_secure_batch_upload_api_accepts_bearer_token(self):
        self.client.force_login(self.lecturer)
        issue_response = self.client.post(reverse("portal:issue_api_token"))
        token = issue_response.json()["result"]["token"]
        self.client.logout()

        upload = SimpleUploadedFile(
            "students.csv",
            self.batch_alias_csv().encode("utf-8"),
            content_type="text/csv",
        )

        response = self.client.post(
            reverse("portal:batch_upload_token_api"),
            {"dataset": upload},
            HTTP_AUTHORIZATION=f"Bearer {token}",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["result"]["processed_count"], 1)

    def test_prediction_api_rejects_invalid_age(self):
        self.client.force_login(self.lecturer)

        response = self.client.post(
            reverse("portal:predict_api"),
            {
                "Age": 30,
                "Gender": 0,
                "ParentalEducation": 2,
                "StudyTimeWeekly": 10,
                "Absences": 4,
                "ParentalSupport": 3,
                "Extracurricular": 1,
                "Sports": 0,
                "Music": 1,
                "Volunteering": 0,
            },
        )

        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertFalse(payload["ok"])
        self.assertIn("Age", payload["errors"])

    def test_batch_upload_api_returns_json_for_valid_csv(self):
        self.client.force_login(self.lecturer)
        upload = SimpleUploadedFile(
            "students.csv",
            self.batch_alias_csv().encode("utf-8"),
            content_type="text/csv",
        )

        response = self.client.post(
            reverse("portal:batch_upload_api"),
            {"dataset": upload},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["result"]["processed_count"], 1)

    def test_batch_upload_api_reuses_active_dataset_mapping_for_custom_headers(self):
        mapping = {
            "StudentID": "student_id",
            "Age": "student_age",
            "Gender": "sex",
            "ParentalEducation": "parent_education",
            "StudyTimeWeekly": "focus_hours",
            "Absences": "absence_count",
            "ParentalSupport": "parent_support",
            "Extracurricular": "clubs",
            "Sports": "athletics",
            "Music": "music",
            "Volunteering": "volunteerwork",
            "GPA": "cgpa",
            "GradeClass": "final_grade",
        }
        self.activate_dataset(
            self.alias_dataset_csv(custom_study_header=True),
            mapping=mapping,
            filename="mapped.csv",
        )
        self.client.force_login(self.lecturer)

        upload = SimpleUploadedFile(
            "batch.csv",
            self.batch_alias_csv(custom_study_header=True).encode("utf-8"),
            content_type="text/csv",
        )
        response = self.client.post(
            reverse("portal:batch_upload_api"),
            {"dataset": upload},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["result"]["processed_count"], 1)

    def test_admin_console_rejects_weak_passwords(self):
        self.client.force_login(self.admin)

        response = self.client.post(
            reverse("portal:admin_console"),
            {
                "action": "add_user",
                "username": "weak-user",
                "password1": "short",
                "password2": "short",
                "role": UserProfile.ROLE_LECTURER,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Password must be at least 8 characters long.")

    def test_user_role_is_stored_in_database(self):
        self.assertEqual(get_user_role(self.admin), UserProfile.ROLE_ADMIN)
        self.assertEqual(get_user_role(self.lecturer), UserProfile.ROLE_LECTURER)