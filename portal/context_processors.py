from .auth_utils import get_user_role


def portal_shell(request):
    current_role = get_user_role(request.user)
    display_username = request.user.username if request.user.is_authenticated else "Guest"
    profile_image_url = ""
    if request.user.is_authenticated:
        try:
            profile_image_url = request.user.profile.profile_image_url
        except Exception:
            profile_image_url = ""
    return {
        "current_role": current_role,
        "display_username": display_username,
        "profile_image_url": profile_image_url,
    }