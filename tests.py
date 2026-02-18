import unittest
import numpy as np
import pandas as pd
import os
from model import StudentPerformanceModel
from auth import UserManager


class TestUserManager(unittest.TestCase):
    """Tests for user authentication and management."""

    def setUp(self):
        self.test_file = "test_users.json"
        self.manager = UserManager(users_file=self.test_file)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_add_user(self):
        success, _ = self.manager.add_user("lecturer1", "password123", "lecturer")
        self.assertTrue(success)
        self.assertIn("lecturer1", self.manager.users)

    def test_add_user_short_password(self):
        success, _ = self.manager.add_user("user1", "abc", "lecturer")
        self.assertFalse(success)

    def test_add_duplicate_user(self):
        self.manager.add_user("user1", "password123", "lecturer")
        success, _ = self.manager.add_user("user1", "password456", "lecturer")
        self.assertFalse(success)

    def test_verify_user(self):
        self.manager.add_user("user1", "password123", "lecturer")
        role = self.manager.verify_user("user1", "password123")
        self.assertEqual(role, "lecturer")

    def test_verify_wrong_password(self):
        self.manager.add_user("user1", "password123", "lecturer")
        role = self.manager.verify_user("user1", "wrongpass")
        self.assertIsNone(role)

    def test_remove_user(self):
        self.manager.add_user("user1", "password123", "lecturer")
        result = self.manager.remove_user("user1")
        self.assertTrue(result)
        self.assertNotIn("user1", self.manager.users)

    def test_cannot_remove_admin(self):
        result = self.manager.remove_user("admin")
        self.assertFalse(result)

    def test_passwords_are_hashed(self):
        self.manager.add_user("user1", "password123", "lecturer")
        stored = self.manager.users["user1"]["password"]
        self.assertNotEqual(stored, "password123")
        self.assertIn("salt", self.manager.users["user1"])


class TestStudentPerformanceModel(unittest.TestCase):
    """Tests for the ML model."""

    def setUp(self):
        self.model = StudentPerformanceModel()
        np.random.seed(42)
        self.X = pd.DataFrame({
            "Age": np.random.randint(15, 19, 100),
            "Gender": np.random.randint(0, 2, 100),
            "ParentalEducation": np.random.randint(0, 5, 100),
            "StudyTimeWeekly": np.random.randint(0, 20, 100),
            "Absences": np.random.randint(0, 30, 100),
            "ParentalSupport": np.random.randint(0, 5, 100),
            "Extracurricular": np.random.randint(0, 2, 100),
            "Sports": np.random.randint(0, 2, 100),
            "Music": np.random.randint(0, 2, 100),
            "Volunteering": np.random.randint(0, 2, 100),
        })
        self.y = pd.Series(np.random.randint(0, 5, 100))

    def test_model_initializes(self):
        self.assertIsNotNone(self.model.scaler)
        self.assertIsNone(self.model.model)
        self.assertFalse(self.model.is_trained)

    def test_train_returns_metrics(self):
        metrics = self.model.train(self.X, self.y)
        for key in ["accuracy", "precision", "recall", "f1_score", "best_model"]:
            self.assertIn(key, metrics)
        for key in ["accuracy", "precision", "recall", "f1_score"]:
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)

    def test_best_model_selected(self):
        self.model.train(self.X, self.y)
        self.assertIsNotNone(self.model.best_model_name)
        self.assertIsNotNone(self.model.comparison_results)
        self.assertIn(self.model.best_model_name, self.model.comparison_results)

    def test_check_imbalance(self):
        info = StudentPerformanceModel.check_imbalance(self.y)
        self.assertIn("distribution", info)
        self.assertIn("ratio", info)
        self.assertIn("is_imbalanced", info)
        self.assertIsInstance(info["is_imbalanced"], bool)

    def test_predict_after_training(self):
        self.model.train(self.X, self.y)
        prediction = self.model.predict(self.X.iloc[[0]])
        self.assertIsNotNone(prediction)
        self.assertIn(prediction[0], range(5))

    def test_predict_before_training_raises(self):
        with self.assertRaises(ValueError):
            self.model.predict(self.X.iloc[[0]])

    def test_predict_grade_label(self):
        self.model.train(self.X, self.y)
        label = self.model.predict_grade_label(self.X.iloc[[0]])
        self.assertIn(label, ["A", "B", "C", "D", "F"])

    def test_feature_importance_stored(self):
        self.model.train(self.X, self.y)
        # Feature importance only available for tree-based models
        if hasattr(self.model.model, "feature_importances_"):
            self.assertIsNotNone(self.model.feature_importance)
            self.assertEqual(len(self.model.feature_importance), 10)

    def test_save_and_load(self):
        """Test model persistence: save to disk and load back."""
        test_path = "test_model.pkl"
        try:
            self.model.train(self.X, self.y)
            self.model.save(test_path)

            loaded = StudentPerformanceModel.load(test_path)
            self.assertIsNotNone(loaded)
            self.assertTrue(loaded.is_trained)
            self.assertEqual(loaded.best_model_name, self.model.best_model_name)

            # Loaded model should produce the same predictions
            orig_pred = self.model.predict(self.X.iloc[[0]])
            loaded_pred = loaded.predict(self.X.iloc[[0]])
            self.assertEqual(orig_pred[0], loaded_pred[0])
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

    def test_save_untrained_raises(self):
        with self.assertRaises(ValueError):
            self.model.save("should_not_exist.pkl")

    def test_load_missing_file_returns_none(self):
        result = StudentPerformanceModel.load("nonexistent_file.pkl")
        self.assertIsNone(result)


class TestIntegration(unittest.TestCase):
    """End-to-end integration tests."""

    def setUp(self):
        self.test_file = "test_users.json"
        self.manager = UserManager(users_file=self.test_file)
        self.model = StudentPerformanceModel()

        np.random.seed(42)
        n = 100
        self.sample_X = pd.DataFrame({
            "Age": np.random.randint(15, 19, n),
            "Gender": np.random.randint(0, 2, n),
            "ParentalEducation": np.random.randint(0, 5, n),
            "StudyTimeWeekly": np.random.randint(0, 20, n),
            "Absences": np.random.randint(0, 30, n),
            "ParentalSupport": np.random.randint(0, 5, n),
            "Extracurricular": np.random.randint(0, 2, n),
            "Sports": np.random.randint(0, 2, n),
            "Music": np.random.randint(0, 2, n),
            "Volunteering": np.random.randint(0, 2, n),
        })
        self.sample_y = pd.Series(np.random.randint(0, 5, n))

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_full_workflow(self):
        """Test: create user -> login -> train model -> predict."""
        # 1. Create user
        success, _ = self.manager.add_user("lecturer1", "password123", "lecturer")
        self.assertTrue(success)

        # 2. Login
        role = self.manager.verify_user("lecturer1", "password123")
        self.assertEqual(role, "lecturer")

        # 3. Train model
        metrics = self.model.train(self.sample_X, self.sample_y)
        self.assertIn("accuracy", metrics)

        # 4. Predict
        prediction = self.model.predict(self.sample_X.iloc[[0]])
        self.assertIsNotNone(prediction)


if __name__ == "__main__":
    unittest.main()
