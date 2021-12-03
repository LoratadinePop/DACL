from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets

from data_aug.view_generator import ContrastiveLearningViewGenerator
from exception.exception import InvalidDatasetSelection


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=0.5):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s,
                                              0.2 * s)
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # GaussianBlur(kernel_size=int(0.1 * size)),
            transforms.ToTensor()
        ])
        return data_transforms

    @staticmethod
    def get_simclr_pipeline_without_transform():
        return transforms.Compose([transforms.ToTensor()])

    def get_gmm_init_dataset(self, name):
        valid_datasets = {
            'cifar10':
            lambda: datasets.CIFAR10(
                self.root_folder,
                train=True,
                transform=self.get_simclr_pipeline_without_transform(),
                download=True),
            'stl10':
            lambda: datasets.STL10(self.root_folder,
                                   split='unlabeled',
                                   transform=self.get_simclr_pipeline_without_transform(),
                                   download=True)
        }
        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()

    '''
    cifar10 dataset: (3,32,32)
    '''
    def get_dataset(self, name, n_views):
        valid_datasets = {
            'cifar10':
            lambda: datasets.CIFAR10(
                self.root_folder,
                train=True,
                transform=ContrastiveLearningViewGenerator(self.get_simclr_pipeline_without_transform(), n_views),
                download=True),
            'stl10':
            lambda: datasets.STL10(self.root_folder,
                                   split='unlabeled',
                                   transform=ContrastiveLearningViewGenerator(self.get_simclr_pipeline_without_transform(), n_views),
                                   download=True)
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()


    def get_dataset_for_simclr_original(self, name, n_views):
        valid_datasets = {
            'cifar10':
            lambda: datasets.CIFAR10(
                self.root_folder,
                train=True,
                transform=ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(32, s=0.5), n_views),
                download=True),
            'stl10':
            lambda: datasets.STL10(self.root_folder,
                                   split='unlabeled',
                                   transform=ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(96), n_views),
                                   download=True)
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
