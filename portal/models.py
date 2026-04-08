import hashlib
import secrets
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone

from .column_mapping import apply_column_mapping


class UserProfile(models.Model):
    ROLE_ADMIN = "admin"
    ROLE_LECTURER = "lecturer"
    ROLE_CHOICES = [
        (ROLE_ADMIN, "Admin"),
        (ROLE_LECTURER, "Lecturer"),
    ]

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="profile",
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default=ROLE_LECTURER)
    profile_image = models.ImageField(upload_to="profiles/", blank=True, null=True)

    def __str__(self):
        return f"{self.user.username} ({self.role})"

    @property
    def profile_image_url(self):
        if not self.profile_image:
            return ""
        return self.profile_image.url


class ApiAccessToken(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="api_token",
    )
    token_prefix = models.CharField(max_length=16)
    token_hash = models.CharField(max_length=64, unique=True)
    created_at = models.DateTimeField(default=timezone.now)
    last_used_at = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        status = "active" if self.is_active else "revoked"
        return f"{self.user.username} API token ({status})"

    @classmethod
    def issue_for_user(cls, user):
        raw_token = f"spp_{secrets.token_urlsafe(32)}"
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

        token, _ = cls.objects.update_or_create(
            user=user,
            defaults={
                "token_prefix": raw_token[:12],
                "token_hash": token_hash,
                "created_at": timezone.now(),
                "last_used_at": None,
                "is_active": True,
            },
        )
        return token, raw_token

    @classmethod
    def resolve_user(cls, raw_token):
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        token = cls.objects.select_related("user").filter(
            token_hash=token_hash,
            is_active=True,
        ).first()

        if token is None:
            return None

        token.last_used_at = timezone.now()
        token.save(update_fields=["last_used_at"])
        return token.user

    def revoke(self):
        self.is_active = False
        self.save(update_fields=["is_active"])


class LecturerDataset(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="datasets",
    )
    file = models.FileField(upload_to="datasets/")
    original_filename = models.CharField(max_length=255)
    column_mapping = models.JSONField(default=dict, blank=True)
    row_count = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=False)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    confirmed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-is_active", "-confirmed_at", "-uploaded_at"]

    def __str__(self):
        status = "active" if self.is_active else "pending"
        return f"{self.user.username} dataset ({status})"

    @property
    def stored_filename(self):
        return Path(self.file.name).name if self.file else self.original_filename

    def load_dataframe(self, apply_mapping=True):
        self.file.open("rb")
        try:
            import pandas as pd

            dataset = pd.read_csv(self.file)
        finally:
            self.file.close()

        if apply_mapping:
            dataset = apply_column_mapping(dataset, self.column_mapping)

        return dataset

    def activate(self):
        type(self).objects.filter(user=self.user, is_active=True).exclude(pk=self.pk).update(
            is_active=False
        )
        self.is_active = True
        self.confirmed_at = timezone.now()
        self.save(update_fields=["is_active", "confirmed_at", "column_mapping", "row_count"])


class InterventionRecord(models.Model):
    class Category(models.TextChoices):
        ATTENDANCE = "attendance", "Attendance"
        STUDY = "study", "Study Habits"
        SUPPORT = "support", "Support"
        ENGAGEMENT = "engagement", "Engagement"
        ACHIEVEMENT = "achievement", "Achievement"
        MONITORING = "monitoring", "Monitoring"

    class Severity(models.TextChoices):
        URGENT = "urgent", "Urgent"
        RECOMMENDED = "recommended", "Recommended"
        OPTIONAL = "optional", "Optional"

    class Status(models.TextChoices):
        PLANNED = "planned", "Planned"
        IN_PROGRESS = "in_progress", "In Progress"
        COMPLETED = "completed", "Completed"
        DISMISSED = "dismissed", "Dismissed"

    class Outcome(models.TextChoices):
        PENDING = "pending", "Pending Review"
        IMPROVED = "improved", "Improved"
        NO_CHANGE = "no_change", "No Meaningful Change"
        WORSENED = "worsened", "Worsened"

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

    SEVERITY_STYLE_MAP = {
        Severity.URGENT: "bg-red-100 text-red-700",
        Severity.RECOMMENDED: "bg-amber-100 text-amber-700",
        Severity.OPTIONAL: "bg-sky-100 text-sky-700",
    }
    STATUS_STYLE_MAP = {
        Status.PLANNED: "bg-surface-container text-on-surface-variant",
        Status.IN_PROGRESS: "bg-sky-100 text-sky-700",
        Status.COMPLETED: "bg-emerald-100 text-emerald-700",
        Status.DISMISSED: "bg-slate-200 text-slate-700",
    }
    OUTCOME_STYLE_MAP = {
        Outcome.PENDING: "bg-surface-container text-on-surface-variant",
        Outcome.IMPROVED: "bg-emerald-100 text-emerald-700",
        Outcome.NO_CHANGE: "bg-amber-100 text-amber-700",
        Outcome.WORSENED: "bg-red-100 text-red-700",
    }

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="intervention_records",
    )
    dataset = models.ForeignKey(
        LecturerDataset,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="intervention_records",
    )
    student_id = models.PositiveIntegerField()
    title = models.CharField(max_length=255)
    category = models.CharField(max_length=24, choices=Category.choices)
    severity = models.CharField(
        max_length=24,
        choices=Severity.choices,
        default=Severity.RECOMMENDED,
    )
    status = models.CharField(
        max_length=24,
        choices=Status.choices,
        default=Status.PLANNED,
    )
    outcome = models.CharField(
        max_length=24,
        choices=Outcome.choices,
        default=Outcome.PENDING,
    )
    target_feature = models.CharField(max_length=64, blank=True)
    feature_value = models.FloatField(null=True, blank=True)
    shap_value = models.FloatField(null=True, blank=True)
    impact_share = models.PositiveIntegerField(default=0)
    predicted_grade = models.CharField(max_length=2, blank=True)
    predicted_risk_label = models.CharField(max_length=24, blank=True)
    predicted_risk_score = models.PositiveIntegerField(default=0)
    note = models.TextField(blank=True)
    outcome_note = models.TextField(blank=True)
    review_date = models.DateField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at", "-created_at"]

    def __str__(self):
        return f"{self.user.username} intervention for student {self.student_id}"

    @property
    def severity_style(self):
        return self.SEVERITY_STYLE_MAP.get(self.severity, "bg-surface-container text-on-surface-variant")

    @property
    def status_style(self):
        return self.STATUS_STYLE_MAP.get(self.status, "bg-surface-container text-on-surface-variant")

    @property
    def outcome_style(self):
        return self.OUTCOME_STYLE_MAP.get(self.outcome, "bg-surface-container text-on-surface-variant")

    @property
    def target_feature_label(self):
        return self.FEATURE_LABELS.get(self.target_feature, self.target_feature or "General")


@receiver(post_save, sender=get_user_model())
def ensure_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(
            user=instance,
            role=UserProfile.ROLE_ADMIN if instance.is_staff else UserProfile.ROLE_LECTURER,
        )