from datasets.cvl.cvl_data_augmentation import CVLDatasetAugmentator
from datasets.cvl.cvl_dataloader import CVLLineDataloader
from datasets.iam.iam_data_augmentation import IAMDatasetAugmentator
from datasets.iam.iam_dataloader import IAMLineDataloader
from datasets.rimes.rimes_data_augmentation import RIMESDatasetAugmentator
from datasets.emnist.emist_data_augmentation import EMNISTDatasetAugmentator
from datasets.rimes.rimes_dataloader import RIMESWordsDataloader
import settings

def augment_datasets():
    emnist_aumentator = EMNISTDatasetAugmentator(
        dataset_path=settings.EMNIST_PATH,
        subfolder_name="lines_png",
        train_split=settings.TRAIN_SPLIT,
        val_split=settings.VAL_SPLIT
    )

    rimes_augmentator = RIMESDatasetAugmentator(
        dataset_path=settings.RIMES_PATH,
        subfolder_name="lines_png",
        train_split=settings.TRAIN_SPLIT,
        val_split=settings.VAL_SPLIT,
        dataloader = RIMESWordsDataloader(settings.RIMES_PATH)
    )

    cvl_augmentator = CVLDatasetAugmentator(
        dataset_path=settings.CVL_PATH,
        subfolder_name="lines_png",
        train_split=settings.TRAIN_SPLIT,
        val_split=settings.VAL_SPLIT,
        dataloader=CVLLineDataloader(settings.CVL_PATH)
    )

    iam_augmentator = IAMDatasetAugmentator(
        dataset_path=settings.IAM_PATH,
        subfolder_name="lines_png",
        train_split=settings.TRAIN_SPLIT,
        val_split=settings.VAL_SPLIT,
        dataloader=IAMLineDataloader(settings.IAM_PATH)
    )

    #emnist_aumentator.augment_dataset()
    #rimes_augmentator.augment_dataset()
    #cvl_augmentator.augment_dataset()
    iam_augmentator.augment_dataset()



if __name__ == "__main__":
    augment_datasets()


