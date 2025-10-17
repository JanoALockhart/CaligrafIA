from abc import ABC
from abc import abstractmethod

class Dataloader(ABC):
    
    @abstractmethod
    def load_samples_tensor(self):
        pass

    @abstractmethod
    def get_vocabulary(self):
        pass

    

        
        







