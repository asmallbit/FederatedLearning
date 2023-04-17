import os
import requests
import shutil
import tarfile
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

class DogCatDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader
        self.url = "https://alist.jwhan99.xyz/d/onedrive/source_code/dog_and_cat/cat_dog.tar.gz"

        if download and not os.path.exists(self.root):    # 下载文件
            self.download_data()

        if self.train:
            self.root = os.path.join(self.root, 'train')
            self.classes = sorted(os.listdir(self.root))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            self.samples = []
            for target in sorted(self.class_to_idx.keys()):
                d = os.path.join(self.root, target)
                for root, _, fnames in sorted(os.walk(d)):
                    for fname in sorted(fnames):
                        if fname.endswith('.jpg'):
                            path = os.path.join(root, fname)
                            item = (path, self.class_to_idx[target])
                            self.samples.append(item)
        else:
            self.root = os.path.join(self.root, 'val')
            self.classes = sorted(os.listdir(self.root))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            self.samples = []
            for target in sorted(self.class_to_idx.keys()):
                d = os.path.join(self.root, target)
                for root, _, fnames in sorted(os.walk(d)):
                    for fname in sorted(fnames):
                        if fname.endswith('.jpg'):
                            path = os.path.join(root, fname)
                            item = (path, self.class_to_idx[target])
                            self.samples.append(item)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def download_file(self):
        print("Downloading the model...")
        r = requests.get(self.url)
        print(f'{self.root}cat_dog.tar.gz')
        with open(f'{self.root}/cat_dog.tar.gz', 'wb') as f:
            f.write(r.content)
        print("Download finished")

    def download_data(self):
        # create the directory if it does not exist
        os.makedirs(self.root, exist_ok=True)
        # download the tar file
        tar_file = os.path.join(self.root, 'cat_dog.tar.gz')
        self.download_file()

        # extract the contents of the tar file
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(self.root)
        
        # 移动文件夹中的所有文件到上级目录
        folder_name = f'{self.root}/{next(os.walk(self.root))[1][0]}'
        for filename in os.listdir(folder_name):
            shutil.move(f'{folder_name}/{filename}', f'{self.root}/{filename}')
        # 删除空文件夹
        os.rmdir(folder_name)

        # remove the tar file
        os.remove(tar_file)
