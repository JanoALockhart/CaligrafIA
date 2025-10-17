from datasets.dataloader import DataLoader
import os
import xml.etree.ElementTree as ET


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