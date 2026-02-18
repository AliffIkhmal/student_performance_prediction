import json
import os
import hashlib
import secrets


class UserManager:
    """Manages user accounts with hashed passwords."""

    def __init__(self, users_file="users.json"):
        self.users_file = users_file
        self.users = self._load_users()

    def _load_users(self):
        """Load users from JSON file. Auto-migrates plain-text passwords to hashed ones."""
        if os.path.exists(self.users_file):
            with open(self.users_file, "r") as f:
                users = json.load(f)

            # Migrate any plain-text passwords to hashed ones
            migrated = False
            for username, data in users.items():
                if "salt" not in data:
                    salt = secrets.token_hex(16)
                    data["salt"] = salt
                    data["password"] = self._hash_password(data["password"], salt)
                    migrated = True

            if migrated:
                self._save_users(users)

            return users

        # No file exists — create default admin account
        salt = secrets.token_hex(16)
        default_users = {
            "admin": {
                "password": self._hash_password("admin123", salt),
                "salt": salt,
                "role": "admin",
            }
        }
        self._save_users(default_users)
        return default_users

    def _hash_password(self, password, salt):
        """Hash a password using SHA-256 with a salt."""
        return hashlib.sha256((salt + password).encode()).hexdigest()

    def _save_users(self, users=None):
        """Save users to JSON file."""
        if users is None:
            users = self.users
        with open(self.users_file, "w") as f:
            json.dump(users, f, indent=4)

    def add_user(self, username, password, role):
        """Add a new user. Returns (success, message)."""
        if len(password) < 6:
            return False, "Password must be at least 6 characters"

        if username in self.users:
            return False, "Username already exists"

        salt = secrets.token_hex(16)
        self.users[username] = {
            "password": self._hash_password(password, salt),
            "salt": salt,
            "role": role,
        }
        self._save_users()
        return True, "User added successfully"

    def remove_user(self, username):
        """Remove a user. Cannot remove the admin account."""
        if username == "admin":
            return False
        if username in self.users:
            del self.users[username]
            self._save_users()
            return True
        return False

    def verify_user(self, username, password):
        """Verify credentials. Returns the user's role, or None if invalid."""
        if username not in self.users:
            return None

        user = self.users[username]
        hashed = self._hash_password(password, user["salt"])

        if hashed == user["password"]:
            return user["role"]
        return None
