from abc import ABC
from abc import abstractmethod

class DatasetBuilder(ABC):

    @abstractmethod
    def get_training_set(self):
        pass

    @abstractmethod
    def get_validation_set(self):
        pass

    @abstractmethod
    def get_test_set(self):
        pass

    @abstractmethod
    def get_vocabulary(self):
        pass