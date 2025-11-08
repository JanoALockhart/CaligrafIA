import datasets.cvl.cvl_data_augmentation as cvl_da
import datasets.rimes.rimes_data_augmentation as rimes_da
import settings

def augment_cvl():
    cvl_da.augment_CVL(
        cvl_path = settings.CVL_PATH,
        name = "lines_png",
        train_split = settings.TRAIN_SPLIT, 
        val_split=settings.VAL_SPLIT, 
    )

def augment_rimes():
    rimes_da.augment_RIMES(
        rimes_path=settings.RIMES_PATH,
        name="lines_png",
        train_split=settings.TRAIN_SPLIT,
        val_split=settings.VAL_SPLIT
    )


if __name__ == "__main__":
    #augment_cvl()
    augment_rimes()