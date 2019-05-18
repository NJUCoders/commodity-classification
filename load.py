import csv
import pickle

from torch.utils.data import Dataset
from torchvision import transforms, utils
from torchvision.datasets.folder import default_loader

train_transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class MyDataSet(Dataset):
    def __init__(self, _csv, root, transform=None, target_transform=None, loader=default_loader):
        with open(_csv, newline='') as fh:
            reader = csv.DictReader(fh)
            imgs = []
            for row in reader:
                fn = root + row['ImageName']
                label = int(row['CategoryId'])
                imgs.append((fn, label))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


def show_batch(imgs):
    grid = utils.make_grid(imgs)
    # plt.imshow(grid.numpy().transpose((1, 2, 0)))
    # plt.title('Batch from dataloader')


def load_from_file():
    train_data = MyDataSet(_csv="easy/data.csv", root="easy/data/", transform=train_transformations)
    # data_loader = DataLoader(train_data, batch_size=100, shuffle=True)
    # print(len(data_loader))
    # for i, (batch_x, batch_y) in enumerate(data_loader):
    #     print(i)
    #     if i < 4:
    #         print(i, batch_x.size(), batch_y.size())
    #         show_batch(batch_x)
    #         plt.axis('off')
    #         plt.show()
    return train_data


class MyDataSet2(Dataset):
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
        label = int(label)
        return img, label

    def __len__(self):
        return len(self.imgs)


def load_from_pickle():
    train_data = MyDataSet2(fn="train_set.pk", transform=transforms.ToTensor())
    return train_data


if __name__ == '__main__':
    train_data = load_from_pickle()

