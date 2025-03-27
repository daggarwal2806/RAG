import unittest
from classes.config_manager import ConfigManager

class TestConfigManager(unittest.TestCase):

    def setUp(self):
        self.config = ConfigManager("config.json")

    def test_get(self):
        self.assertEqual(self.config.get("embedding_model_name"), "sentence-transformers/all-mpnet-base-v2")

    def test_get_nonexistent_key(self):
        self.assertIsNone(self.config.get("nonexistent_key"))

if __name__ == "__main__":
    unittest.main()