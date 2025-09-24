from abc import ABC
from abc import abstractmethod

class DataLoader(ABC):
    
    @abstractmethod
    def load_samples_tensor(self):
        pass

class IAMLineDataloader(DataLoader):
    """Loads the image paths and its ground truths"""
    def __init__(self, path_to_IAM):
        self.path_to_IAM = path_to_IAM
        self.samples_path = path_to_IAM + '/lines/'
        self.gt_file_path = path_to_IAM + '/ascii/lines.txt'
        self.alphabet = set()
    
    def load_samples_tensor(self):
        samples = []
        labels = []
        gt_file = open(self.gt_file_path)

        for line in gt_file:
            if line and line[0] != '#':
                line_split = line.strip().split(' ')
                img_name = line_split[0]
                img_path = self._parse_img_path(img_name)
                samples.append(img_path)
                label = line_split[8].replace('|', ' ')
                labels.append(label)

        return (samples, labels)
    
    def _parse_img_path(self, img_name:str):
        name_split = img_name.split('-')
        first_dir = name_split[0]
        second_dir = f'{name_split[0]}-{name_split[1]}'
        file_path = f"{self.path_to_IAM}/lines/{first_dir}/{second_dir}/{img_name}.png" 

        return file_path




