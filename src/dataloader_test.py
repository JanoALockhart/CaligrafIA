import settings
from dataloader import IAMLineDataloader

def main():
    iam_dataloader = IAMLineDataloader(settings.IAM_PATH)
    (samples, labels) = iam_dataloader.load_samples_tensor()
    print(samples)
    print(labels)


if __name__ == "__main__":
    main()