from django.contrib.auth import get_user_model

from .models import ApiAccessToken, UserProfile


def get_user_role(user):
    if not getattr(user, "is_authenticated", False):
        return None

    try:
        return user.profile.role
    except UserProfile.DoesNotExist:
        profile = UserProfile.objects.create(
            user=user,
            role=UserProfile.ROLE_ADMIN if user.is_staff else UserProfile.ROLE_LECTURER,
        )
        return profile.role


def set_user_role(user, role):
    profile, _ = UserProfile.objects.get_or_create(user=user)
    profile.role = role
    profile.save(update_fields=["role"])

    is_admin = role == UserProfile.ROLE_ADMIN
    updated_fields = []
    if user.is_staff != is_admin:
        user.is_staff = is_admin
        updated_fields.append("is_staff")
    if updated_fields:
        user.save(update_fields=updated_fields)

    return profile


def bootstrap_default_admin():
    user_model = get_user_model()
    if user_model.objects.exists():
        return

    admin_user = user_model.objects.create_user(
        username="admin",
        password="admin123",
        is_staff=True,
    )
    set_user_role(admin_user, UserProfile.ROLE_ADMIN)


def issue_api_token(user):
    return ApiAccessToken.issue_for_user(user)


def revoke_api_token(user):
    token = ApiAccessToken.objects.filter(user=user, is_active=True).first()
    if token is None:
        return False

    token.revoke()
    return True


def get_api_token_summary(user):
    if not getattr(user, "is_authenticated", False):
        return {
            "has_token": False,
            "token_prefix": None,
            "created_at": None,
            "last_used_at": None,
        }

    token = ApiAccessToken.objects.filter(user=user, is_active=True).first()
    if token is None:
        return {
            "has_token": False,
            "token_prefix": None,
            "created_at": None,
            "last_used_at": None,
        }

    return {
        "has_token": True,
        "token_prefix": token.token_prefix,
        "created_at": token.created_at.strftime("%Y-%m-%d %H:%M"),
        "last_used_at": token.last_used_at.strftime("%Y-%m-%d %H:%M") if token.last_used_at else "Never",
    }


def extract_api_token(request):
    auth_header = request.META.get("HTTP_AUTHORIZATION", "").strip()
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()

    custom_header = request.META.get("HTTP_X_API_TOKEN", "").strip()
    if custom_header:
        return custom_header

    return None


def authenticate_api_token(raw_token):
    if not raw_token:
        return None

    return ApiAccessToken.resolve_user(raw_token)
