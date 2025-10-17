from abc import ABC
from abc import abstractmethod

class DataLoader(ABC):
    
    @abstractmethod
    def load_samples_tensor(self):
        pass

    @abstractmethod
    def get_vocabulary(self):
        pass
    
    

        
        







