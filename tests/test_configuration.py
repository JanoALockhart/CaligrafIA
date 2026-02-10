import unittest
from unittest import mock
from src.configuration import ConfigurationLoader

class TestConfigurationLoader(unittest.TestCase):

    def test_setup_configurations(self):
        #Arrange
        mock_config_content = """
            preprocessFolder: "prepFolder"
            datasets:
            - name: "ds_name_1"
              path: "/path/to/ds"
              splits:
                train: 0.6
                validation: 0.2
                test: 0.2
            """
        
        file_mock = unittest.mock.mock_open(read_data=mock_config_content)
        
        with unittest.mock.patch("builtins.open", file_mock) as mock:
            #Act
            config = ConfigurationLoader.load_configurations()
        
        #Assert
        self.assertEqual(config.preprocessFolder, "prepFolder")
        self.assertEqual(len(config.datasets), 1)
        self.assertEqual(config.datasets[0].name, "ds_name_1")
        self.assertEqual(config.datasets[0].path, "/path/to/ds")
        self.assertEqual(config.datasets[0].train_percentage, 0.6)
        self.assertEqual(config.datasets[0].validation_percentage, 0.2)
        self.assertEqual(config.datasets[0].test_percentage, 0.2)

if __name__ == "__main__":
    unittest.main()

