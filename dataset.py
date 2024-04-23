'''
Attempt to create my simple implementation for dataset
'''
import logging
from pathlib import Path
import random
import copy
from torchvision.io import read_image
import torchvision.transforms as transforms
import torch


logger = logging.getLogger()


class Dataset:
    CANCERS = ['melanoma', 'nevus', 'seborrheic_keratosis']
    DATASETS = ['test', 'train', 'valid']

    def __init__(
            self,
            path_to_dataset,
            path_to_cache,
            preprocessor,
            batch_size,
            recreate_cache=False):
        logger.info(f'Preprocessor: {preprocessor}')
        logger.info(f'Path to dataset: {path_to_dataset}')
        logger.info(f'Path to cache: {path_to_cache}')

        self._preprocessor = preprocessor
        if recreate_cache:
            self._preprocess_imgs(path_to_dataset, path_to_cache)
        self._path = path_to_cache

        self._batch_size = batch_size

        # initialize with 0
        self._imgs = {}
        for dataset in self.DATASETS:
            self._imgs[dataset] = {}
            for cancer in self.CANCERS:
                self._imgs[dataset][cancer] = self._collect_images(dataset, cancer)

        logger.debug(f'Images: {self._imgs}')

    def train(self):
        return self._get_random_imgs('train')

    def test(self):
        return self._get_random_imgs('test')

    def validate(self):
        return self._get_random_imgs('valid')

    def _get_random_imgs(self, dataset):
        imgs = copy.deepcopy(self._imgs[dataset])
        cancers = copy.deepcopy(self.CANCERS)
        batch = []
        ground_truth = []
        remaining_size = self._batch_size
        while True:
            logger.debug(f'remaining_size: {remaining_size}')
            if not cancers:
                break
            if remaining_size > 0:
                cancer = random.sample(cancers, 1)[0]
                try:
                    img = random.sample(imgs[cancer], 1)[0]
                    batch.append(str(img))
                    ground_truth.append(self.CANCERS.index(cancer))
                    remaining_size -= 1
                    imgs[cancer].remove(img)
                    if not len(imgs[cancer]):
                        cancers.remove(cancer)
                except ValueError as e:
                    pass
            else:
                yield (self._prep_batch(batch), ground_truth)
                batch = []
                ground_truth = []
                remaining_size = self._batch_size
        yield (self._prep_batch(batch), ground_truth)

    def _prep_batch(self, batch):
        batch = [read_image(str(file)).unsqueeze(0) for file in batch]
        batch = torch.cat(batch, dim=0)
        return batch.float()

    def _collect_images(self, dataset, cancer):
        res = []
        path = Path(self._path)/Path(dataset)/Path(cancer)
        files = path.glob("**/*")
        for file in files:
            if file.is_file():
                res.append(file)
        return res

    def _preprocess_imgs(self, path_to_dataset, path_to_cache):
        for dataset in self.DATASETS:
            for cancer in self.CANCERS:
                path = Path(path_to_dataset)/Path(dataset)/Path(cancer)
                files = path.glob("**/*")
                for file in files:
                    if file.is_file():
                        tensor_new = self._preprocessor(
                            read_image(str(file))).unsqueeze(0)
                        img_new = transforms.ToPILImage()(tensor_new.squeeze(0))
                        new_dir = Path(path_to_cache)/Path(dataset)/Path(cancer)
                        new_dir.mkdir(parents=True, exist_ok=True)
                        img_new.save(new_dir/file.name)
        

def main():
    logging.basicConfig(level=logging.INFO)
    path_to_dataset = ''
    path_to_cache = ''
    from torchvision.models import ResNet50_Weights
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()

    dataset = Dataset(
        path_to_dataset,
        path_to_cache,
        preprocess,
        30,
        recreate_cache=False)
    import time
    start_time = time.time()
    for i in dataset.test():
        print(f'main loop: {len(i)}')
    print(f'Duration: {time.time() - start_time}')
    start_time = time.time()
    for i in dataset.test():
        print(f'main loop: {len(i)}')
    print(f'Duration: {time.time() - start_time}')

if __name__ == '__main__':
    main()
