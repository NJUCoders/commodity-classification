import pickle

from torch.utils.data import Dataset
from torchvision import transforms, utils

# train_transformations = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.Resize((32, 32)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

train_transformations = transforms.Compose([
    transforms.ToTensor()
])


class PickleDataSet(Dataset):
    def __init__(self, fn, transform=None, target_transform=None):
        with open(fn, "rb") as f:
            pk = pickle.load(f)
            self.imgs = list(zip(pk["image"], pk["label"]))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, label = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = int(label) - 1
        return img, label

    def __len__(self):
        return len(self.imgs)


def load_from_pickle():
    train_data = PickleDataSet(fn="train_set.pk", transform=train_transformations)
    return train_data


if __name__ == '__main__':
    train_data = load_from_pickle()

