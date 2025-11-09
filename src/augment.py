
from datasets.cvl.cvl_data_augmentation import augment_CVL
from datasets.rimes.rimes_data_augmentation import augment_RIMES
from datasets.emnist.emist_data_augmentation import EMNISTDatasetAugmentator
import settings

def augment_cvl():
    augment_CVL(
        cvl_path = settings.CVL_PATH,
        name = "lines_png",
        train_split = settings.TRAIN_SPLIT, 
        val_split=settings.VAL_SPLIT, 
    )

def augment_rimes():
    augment_RIMES(
        rimes_path=settings.RIMES_PATH,
        name="lines_png",
        train_split=settings.TRAIN_SPLIT,
        val_split=settings.VAL_SPLIT
    )


def augment_datasets():
    emnist_aumentator = EMNISTDatasetAugmentator(
        dataset_path=settings.EMNIST_PATH,
        subfolder_name="lines_png",
        train_split=settings.TRAIN_SPLIT,
        val_split=settings.VAL_SPLIT
    )


    emnist_aumentator.augment_dataset()

if __name__ == "__main__":
    augment_datasets()


