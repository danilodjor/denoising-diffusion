from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset


def transform(examples, img_size):
    preprocess = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


def get_data(config):
    img_size = config["data"]["img_size"]
    batch_size = config["training"]["batch_size"]
    dataset = config["data"]["dataset"]

    dataset = load_dataset(dataset, split="train")

    dataset.set_transform(lambda x: transform(x, img_size))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
