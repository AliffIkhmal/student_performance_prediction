from django import forms
from django.contrib.auth import authenticate, get_user_model
from django.contrib.auth.forms import UserCreationForm
from django.core.exceptions import ValidationError
from django.utils import timezone

from model import StudentPerformanceModel

from .auth_utils import get_user_role, set_user_role
from .column_mapping import build_mapping_rows
from .models import InterventionRecord, UserProfile
from .services import EDUCATION_LABELS, SUPPORT_LABELS, get_form_defaults


FORM_FIELD_CLASS = "form-field"
LOGIN_FIELD_CLASS = (
    "w-full rounded-2xl border border-outline-variant/50 bg-surface-container-low "
    "py-3 pl-12 pr-4 text-sm shadow-sm outline-none transition "
    "focus:border-primary focus:bg-white focus:ring-2 focus:ring-primary/10"
)


def form_field_attrs(**kwargs):
    attrs = {"class": FORM_FIELD_CLASS}
    attrs.update(kwargs)
    return attrs


def login_field_attrs(**kwargs):
    attrs = {"class": LOGIN_FIELD_CLASS}
    attrs.update(kwargs)
    return attrs


def password_field_attrs(**kwargs):
    attrs = form_field_attrs(style="padding-right: 3.5rem;")
    attrs.update(kwargs)
    return attrs


def login_password_field_attrs(**kwargs):
    attrs = login_field_attrs(style="padding-right: 3.5rem;")
    attrs.update(kwargs)
    return attrs


class PortalLoginForm(forms.Form):
    username = forms.CharField(
        max_length=150,
        widget=forms.TextInput(
            attrs=login_field_attrs(placeholder="admin or lecturer username")
        ),
    )
    password = forms.CharField(
        strip=False,
        widget=forms.PasswordInput(
            attrs=login_password_field_attrs(placeholder="Enter your password")
        ),
    )
    keep_session_active = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(
            attrs={
                "class": "rounded border-outline-variant text-primary focus:ring-primary",
            }
        ),
    )

    def __init__(self, request=None, *args, **kwargs):
        self.request = request
        self.user_cache = None
        super().__init__(*args, **kwargs)

    def clean(self):
        cleaned_data = super().clean()
        username = cleaned_data.get("username")
        password = cleaned_data.get("password")

        if username and password:
            self.user_cache = authenticate(
                self.request,
                username=username,
                password=password,
            )
            if self.user_cache is None:
                raise ValidationError("Invalid username or password.")
            if not self.user_cache.is_active:
                raise ValidationError("This account is disabled.")

        return cleaned_data

    def get_user(self):
        return self.user_cache


class StudentPredictionForm(forms.Form):
    YES_NO_CHOICES = [(0, "No"), (1, "Yes")]
    GENDER_CHOICES = [(0, "Male"), (1, "Female")]
    EDUCATION_CHOICES = list(enumerate(EDUCATION_LABELS))
    SUPPORT_CHOICES = list(enumerate(SUPPORT_LABELS))

    Age = forms.IntegerField(
        min_value=15,
        max_value=18,
        widget=forms.NumberInput(form_field_attrs(min=15, max=18)),
    )
    Gender = forms.TypedChoiceField(
        coerce=int,
        choices=GENDER_CHOICES,
        widget=forms.Select(form_field_attrs()),
    )
    ParentalEducation = forms.TypedChoiceField(
        coerce=int,
        choices=EDUCATION_CHOICES,
        widget=forms.Select(form_field_attrs()),
    )
    StudyTimeWeekly = forms.FloatField(
        min_value=0,
        max_value=20,
        widget=forms.NumberInput(form_field_attrs(min=0, max=20, step="0.1")),
    )
    Absences = forms.IntegerField(
        min_value=0,
        max_value=30,
        widget=forms.NumberInput(form_field_attrs(min=0, max=30)),
    )
    ParentalSupport = forms.TypedChoiceField(
        coerce=int,
        choices=SUPPORT_CHOICES,
        widget=forms.Select(form_field_attrs()),
    )
    Extracurricular = forms.TypedChoiceField(
        coerce=int,
        choices=YES_NO_CHOICES,
        widget=forms.Select(form_field_attrs()),
    )
    Sports = forms.TypedChoiceField(
        coerce=int,
        choices=YES_NO_CHOICES,
        widget=forms.Select(form_field_attrs()),
    )
    Music = forms.TypedChoiceField(
        coerce=int,
        choices=YES_NO_CHOICES,
        widget=forms.Select(form_field_attrs()),
    )
    Volunteering = forms.TypedChoiceField(
        coerce=int,
        choices=YES_NO_CHOICES,
        widget=forms.Select(form_field_attrs()),
    )

    def __init__(self, *args, user=None, **kwargs):
        initial = kwargs.setdefault("initial", {})
        defaults = get_form_defaults(user=user)
        for field_name, value in defaults.items():
            initial.setdefault(field_name, value)
        super().__init__(*args, **kwargs)

    def to_feature_payload(self):
        return {
            feature: self.cleaned_data[feature]
            for feature in StudentPerformanceModel.FEATURES
        }


class BatchUploadForm(forms.Form):
    MAX_UPLOAD_SIZE = 5 * 1024 * 1024

    dataset = forms.FileField(
        widget=forms.ClearableFileInput(
            attrs={
                "accept": ".csv",
                "class": "sr-only",
            }
        )
    )

    def clean_dataset(self):
        dataset = self.cleaned_data["dataset"]

        if not dataset.name.lower().endswith(".csv"):
            raise ValidationError("Upload a CSV file with a .csv extension.")
        if dataset.size == 0:
            raise ValidationError("The uploaded file is empty.")
        if dataset.size > self.MAX_UPLOAD_SIZE:
            raise ValidationError("The uploaded file is too large. Keep it under 5 MB.")

        return dataset


class DatasetUploadForm(BatchUploadForm):
    MAX_UPLOAD_SIZE = 10 * 1024 * 1024


class DatasetMappingForm(forms.Form):
    def __init__(self, *args, csv_columns=None, selected_mapping=None, canonical_columns=None, required_columns=None, **kwargs):
        self.mapping_rows = build_mapping_rows(
            csv_columns or [],
            selected_mapping=selected_mapping,
            canonical_columns=canonical_columns,
            required_columns=required_columns,
        )
        super().__init__(*args, **kwargs)

        for row in self.mapping_rows:
            field_name = self.field_name(row["canonical"])
            choices = [("", "Not mapped")] + [
                (column_name, column_name)
                for column_name in row["choices"]
            ]
            self.fields[field_name] = forms.ChoiceField(
                required=False,
                choices=choices,
                initial=row["selected_actual"],
                widget=forms.Select(form_field_attrs()),
            )
            row["field_name"] = field_name

    @staticmethod
    def field_name(canonical_name):
        return f"map_{canonical_name}"

    def clean(self):
        cleaned_data = super().clean()
        selected_columns = {}
        column_mapping = {}

        for row in self.mapping_rows:
            field_name = row["field_name"]
            actual_name = (cleaned_data.get(field_name) or "").strip()

            if row["required"] and not actual_name:
                self.add_error(field_name, f"Map a CSV column to {row['label']}.")
                continue

            if not actual_name:
                continue

            if actual_name in selected_columns:
                self.add_error(
                    field_name,
                    f"{actual_name} is already assigned to {selected_columns[actual_name]}.",
                )
                continue

            selected_columns[actual_name] = row["label"]
            column_mapping[row["canonical"]] = actual_name

        cleaned_data["column_mapping"] = column_mapping
        return cleaned_data


class InterventionCreateForm(forms.Form):
    title = forms.CharField(widget=forms.HiddenInput())
    category = forms.ChoiceField(
        choices=InterventionRecord.Category.choices,
        widget=forms.HiddenInput(),
    )
    severity = forms.ChoiceField(
        choices=InterventionRecord.Severity.choices,
        widget=forms.HiddenInput(),
    )
    target_feature = forms.CharField(required=False, widget=forms.HiddenInput())
    feature_value = forms.FloatField(required=False, widget=forms.HiddenInput())
    shap_value = forms.FloatField(required=False, widget=forms.HiddenInput())
    impact_share = forms.IntegerField(required=False, widget=forms.HiddenInput())
    predicted_grade = forms.CharField(max_length=2, widget=forms.HiddenInput())
    predicted_risk_label = forms.CharField(max_length=24, widget=forms.HiddenInput())
    predicted_risk_score = forms.IntegerField(widget=forms.HiddenInput())
    note = forms.CharField(
        required=False,
        widget=forms.Textarea(
            attrs=form_field_attrs(
                rows=3,
                placeholder="Optional note for the planned follow-up",
            )
        ),
    )
    review_date = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs=form_field_attrs(type="date")),
    )

    def clean_target_feature(self):
        target_feature = (self.cleaned_data.get("target_feature") or "").strip()
        if target_feature and target_feature not in StudentPerformanceModel.FEATURES:
            raise ValidationError("Invalid target feature.")
        return target_feature


class InterventionOutcomeForm(forms.Form):
    status = forms.ChoiceField(
        choices=InterventionRecord.Status.choices,
        widget=forms.Select(form_field_attrs()),
    )
    outcome = forms.ChoiceField(
        choices=InterventionRecord.Outcome.choices,
        widget=forms.Select(form_field_attrs()),
    )
    outcome_note = forms.CharField(
        required=False,
        widget=forms.Textarea(
            attrs=form_field_attrs(
                rows=3,
                placeholder="Record what changed after the intervention",
            )
        ),
    )
    review_date = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs=form_field_attrs(type="date")),
    )


class InterventionHistoryFilterForm(forms.Form):
    student_id = forms.IntegerField(
        required=False,
        min_value=1,
        widget=forms.NumberInput(form_field_attrs(placeholder="Student ID")),
    )
    category = forms.ChoiceField(
        required=False,
        choices=[("", "All categories"), *InterventionRecord.Category.choices],
        widget=forms.Select(form_field_attrs()),
    )
    severity = forms.ChoiceField(
        required=False,
        choices=[("", "All severities"), *InterventionRecord.Severity.choices],
        widget=forms.Select(form_field_attrs()),
    )
    status = forms.ChoiceField(
        required=False,
        choices=[("", "All statuses"), *InterventionRecord.Status.choices],
        widget=forms.Select(form_field_attrs()),
    )
    outcome = forms.ChoiceField(
        required=False,
        choices=[("", "All outcomes"), *InterventionRecord.Outcome.choices],
        widget=forms.Select(form_field_attrs()),
    )
    target_feature = forms.ChoiceField(
        required=False,
        choices=[("", "All target features")] + [
            (feature_name, feature_name)
            for feature_name in StudentPerformanceModel.FEATURES
        ],
        widget=forms.Select(form_field_attrs()),
    )
    date_from = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs=form_field_attrs(type="date")),
    )
    date_to = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs=form_field_attrs(type="date")),
    )

    def clean(self):
        cleaned_data = super().clean()
        date_from = cleaned_data.get("date_from")
        date_to = cleaned_data.get("date_to")
        if date_from and date_to and date_from > date_to:
            self.add_error("date_to", "End date must be on or after the start date.")
        return cleaned_data


class ProfileImageUploadForm(forms.Form):
    MAX_UPLOAD_SIZE = 3 * 1024 * 1024
    ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

    profile_image = forms.ImageField(
        required=True,
        widget=forms.FileInput(attrs={"accept": "image/png,image/jpeg,image/webp,image/gif"}),
    )

    def clean_profile_image(self):
        profile_image = self.cleaned_data["profile_image"]
        content_type = getattr(profile_image, "content_type", "")
        if content_type and content_type not in self.ALLOWED_CONTENT_TYPES:
            raise ValidationError("Upload a JPG, PNG, WEBP, or GIF image.")
        if profile_image.size > self.MAX_UPLOAD_SIZE:
            raise ValidationError("Keep the profile image under 3 MB.")
        return profile_image


class AdminUserCreateForm(UserCreationForm):
    role = forms.ChoiceField(
        choices=UserProfile.ROLE_CHOICES,
        widget=forms.Select(form_field_attrs()),
    )

    class Meta(UserCreationForm.Meta):
        model = get_user_model()
        fields = ("username",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["username"].widget.attrs.update(form_field_attrs())
        self.fields["password1"].widget.attrs.update(password_field_attrs())
        self.fields["password2"].widget.attrs.update(password_field_attrs())
        self.fields["username"].help_text = ""
        self.fields["password1"].help_text = "Use at least 8 characters for a safer account."
        self.fields["password2"].help_text = "Re-enter the password to confirm it."

    def clean_username(self):
        username = self.cleaned_data["username"].strip()
        if get_user_model().objects.filter(username=username).exists():
            raise ValidationError("Username already exists.")
        return username

    def clean_password1(self):
        password = self.cleaned_data["password1"]
        if len(password) < 8:
            raise ValidationError("Password must be at least 8 characters long.")
        return password

    def save(self, commit=True):
        user = super().save(commit=False)
        role = self.cleaned_data["role"]
        user.is_staff = role == UserProfile.ROLE_ADMIN
        if commit:
            user.save()
            set_user_role(user, role)
        return user


class AdminUserRemoveForm(forms.Form):
    username = forms.CharField(widget=forms.HiddenInput())

    def __init__(self, *args, current_user=None, **kwargs):
        self.current_user = current_user
        self.target_user = None
        super().__init__(*args, **kwargs)

    def clean_username(self):
        username = self.cleaned_data["username"].strip()
        user_model = get_user_model()
        target = user_model.objects.filter(username=username).first()

        if target is None:
            raise ValidationError("The selected user no longer exists.")
        if self.current_user is not None and target.pk == self.current_user.pk:
            raise ValidationError("You cannot remove the account you are currently using.")
        if (
            get_user_role(target) == UserProfile.ROLE_ADMIN
            and user_model.objects.filter(profile__role=UserProfile.ROLE_ADMIN).count() <= 1
        ):
            raise ValidationError("At least one admin account must remain in the system.")

        self.target_user = target
        return username

    def save(self):
        if self.target_user is None:
            raise ValueError("Cannot remove a user before validation completes.")
        self.target_user.delete()
