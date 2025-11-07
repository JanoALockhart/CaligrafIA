import datasets.cvl.cvl_data_augmentation as cvl_da
import settings

def augment():
    cvl_da.augment_CVL(
        cvl_path = settings.CVL_PATH,
        name = "lines_png",
        train_split = settings.TRAIN_SPLIT, 
        val_split=settings.VAL_SPLIT, 
        augmented = False
    )



if __name__ == "__main__":
    augment()