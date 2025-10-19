from datasets.dataloader import Dataloader

class RIMESWordsDataloader(Dataloader):
    def __init__(self, path_to_RIMES):
        self.path_to_RIMES = path_to_RIMES
        self.gt_file_path = path_to_RIMES + '/imagettes_mots_cursif/goodSnippets_total/goodSnippets_total.dat'
        self.vocabulary = set()

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
                self.vocabulary.update(label)

        return (samples, labels)
    
    def get_vocabulary(self):
        return self.vocabulary
    