from abc import ABC
from abc import abstractmethod
import os
import xml.etree.ElementTree as ET

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
    
class CVLLineDataloader(DataLoader):
    def __init__(self, path_to_CVL):
        self.path_to_CVL = path_to_CVL
        self.test_samples_path = path_to_CVL + '/cvl-database-1-1/testset/lines/'
        self.gt_test_file_path = path_to_CVL + '/cvl-database-1-1/testset/xml/'
        self.train_samples_path = path_to_CVL + '/cvl-database-1-1/trainset/lines/'
        self.gt_train_file_path = path_to_CVL + '/cvl-database-1-1/trainset/xml/'
        self.namespace = {'pg' : "http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19"}

    def load_samples_tensor(self):
        samples = []
        labels = []

        train_samples, train_labels = self._build_sample_gt_pairs(self.train_samples_path, self.gt_train_file_path)
        samples.extend(train_samples)
        labels.extend(train_labels)

        test_samples, test_labels = self._build_sample_gt_pairs(self.test_samples_path, self.gt_test_file_path)
        samples.extend(test_samples)
        labels.extend(test_labels)

        return (samples, labels)

    def _build_sample_gt_pairs(self, samples_path, gt_file_path):
        samples = []
        labels = []

        gt_files = [os.path.join(gt_file_path, file) for file in sorted(os.listdir(gt_file_path)) if file.endswith('.xml')]

        for file_path in gt_files:
            try:
                paths, phrases = self._process_xml(samples_path, file_path)
                samples.extend(paths)
                labels.extend(phrases)
            except UnicodeDecodeError:
                print(f"ENCODING ERROR in {file_path}. Ignoring file...")

        return samples, labels

    def _process_xml(self, samples_path, file_path):
        paths = []
        labels = []

        with open(file_path, "r", errors="ignore") as f:
            content = f.read()
        root = ET.fromstring(content)
        
        page_list = root.findall(".//pg:AttrRegion[@attrType='3'][@fontType='2']", self.namespace)
        if len(page_list) > 0:
            page = page_list[0]
            for phrase in page.findall("pg:AttrRegion", self.namespace):
                img_path = self._build_img_path(samples_path, phrase)
                label = self._build_label(phrase)
                paths.append(img_path)
                labels.append(label)
            
        return paths, labels
    
    def _build_img_path(self, samples_path, phrase):
        img_name = phrase.get("id")
        number_folder = img_name.split("-")[0]
        path = samples_path + number_folder + "/" + img_name + ".tif"
        
        return path 

    def _build_label(self, phrase):
        words = [word.get("text") for word in phrase.findall("pg:AttrRegion", self.namespace) if word.get("text") is not None]
        return " ".join(words)


class RIMESWordsDataloader(DataLoader):
    def __init__(self, path_to_RIMES):
        self.path_to_RIMES = path_to_RIMES
        self.gt_file_path = path_to_RIMES + '/imagettes_mots_cursif/goodSnippets_total/goodSnippets_total.dat'

    def load_samples_tensor(self):
        samples = []
        labels = []

        with open(self.gt_file_path, "r") as gt_file:
            for line in gt_file:
                inverted_bar_split = line.strip().split('\\')
                bar_split = inverted_bar_split[2].split("/") 
                lot = bar_split[0]
                lot = lot + "_rimes_version_definitive"
                folder = bar_split[1]
                space_split = bar_split[2].split(" ")
                img_file_name = space_split[0]
                
                path = f"{self.path_to_RIMES}/imagettes_mots_cursif/{lot}/{folder}/{img_file_name}" 
                label = space_split[1]

                samples.append(path)
                labels.append(label)

        print(len(samples))

        return (samples, labels)
    

        
        







