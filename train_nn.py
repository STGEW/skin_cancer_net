import argparse
from pathlib import Path
import logging
import importlib.util
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import time

from dataset import Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


DEF_PREPROCESSOR = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

fm = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=fm)
file_handler = logging.FileHandler('training.log')
file_handler.setFormatter(logging.Formatter(fm))
logger = logging.getLogger()
logger.addHandler(file_handler)


COUNT_OF_CANCER_TYPES = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Python script responsible for training neural network for '
        'skin cancer detection')

    parser.add_argument(
        '--dataset_path',
        type=Path,
        help='Path to dataset. Should be a directory with 3 directories inside: '
        """
            ├── test
            │   ├── melanoma
            │   │  
            │   ├── nevus
            │   │   
            │   ├── seborrheic_keratosis
            │   │   
            ├── train
            │   ├── melanoma
            │   │  
            │   ├── nevus
            │   │   
            │   ├── seborrheic_keratosis
            │   │   
            └── valid
                ├── melanoma
                │   
                ├── nevus
                │  
                ├── seborrheic_keratosis
        Inside directories should be jpg files
        """)

    parser.add_argument(
        '--dataset_cache_path',
        type=Path,
        help='''
        Path to cache for dataset. Neural network will do a preprocessing
        and convert dataset to images appropriate for analysis. It allows us to 
        save a lot of time in the future''')

    parser.add_argument(
        '--recreate_cache',
        type=bool,
        help='''
        True if you want to recreate preprocessed cache.
        False if you want to reuse existing''')

    parser.add_argument(
        '--load_existing_model',
        type=bool,
        help='''
        True if you want to load the model from disk.
        False to start from scratch''')

    parser.add_argument(
        '--model_file',
        type=Path,
        help='Path to the python file with neural network architecture')

    parser.add_argument(
        '--model_weights',
        type=Path,
        help='Path to the weights of neural network')

    parser.add_argument(
        '--batch_size',
        type=int,
        help='count of images in a batch')

    parser.add_argument(
        '--epochs_count',
        type=int,
        help='count of epochs for training')

    parser.add_argument(
        '--learning_rate',
        type=float,
        help='count of epochs for training')

    parser.add_argument(
        '--save_every_epochs',
        type=int,
        help='count of epochs when to save nn')

    parser.add_argument(
        '--evaluate_epochs_period',
        type=int,
        help="every evaluate_epochs_period we'll evaluate accuracy with a validation set")

    args = parser.parse_args()
    logger.info(args)
    return args


def import_model(path_to_model):
    logger.info(
        f'Import Model class from path: "{path_to_model}"')
    spec = importlib.util.spec_from_file_location('Model', path_to_model)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Model(COUNT_OF_CANCER_TYPES)
    return model


def evaluate_accuracy(model, dataloader):

    total_correct = 0
    total_samples = 0
    total_loss = 0
    count = 0

    model.eval()
    categories_acc = {}

    with torch.inference_mode():
        for batch, ground_truth in dataloader:

            count += len(batch)
            batch = batch.to(device)

            ground_truth = torch.tensor(ground_truth)
            
            ground_truth = F.one_hot(
                ground_truth, num_classes=COUNT_OF_CANCER_TYPES)
            ground_truth = ground_truth.to(torch.float32)
            ground_truth = ground_truth.to(device)

            prediction = model(batch)
            loss = criterion(prediction, ground_truth)
            total_loss += loss.item()


            prediction = F.softmax(prediction, dim=1)

            _, predicted = torch.max(prediction, 1)
            _, ground_truth = torch.max(ground_truth, 1)
            correct_guess = predicted == ground_truth

            for true_cat, result in zip(ground_truth, correct_guess):
                cat = true_cat.item()
                if cat not in categories_acc:
                    categories_acc[cat] = {'total': 0, 'correct': 0}
                if result:
                    categories_acc[cat]['correct'] += 1
                categories_acc[cat]['total'] += 1
            total_correct += (predicted == ground_truth).sum().item()
            total_samples += ground_truth.size(0)

    avg_loss = total_loss / count
    accuracy = 100 * total_correct / count
    return avg_loss, accuracy, categories_acc
    

def main():
    logging.basicConfig(level=logging.INFO)

    args = parse_arguments()

    dataset_path = args.dataset_path
    dataset_cache_path = args.dataset_cache_path
    recreate_cache = args.recreate_cache
    model_file = args.model_file
    model_weights = args.model_weights
    load_existing_model = args.load_existing_model
    batch_size = args.batch_size
    epochs_count = args.epochs_count
    learning_rate = args.learning_rate
    save_every_epochs = args.save_every_epochs
    evaluate_epochs_period = args.evaluate_epochs_period

    model = import_model(model_file)
    model.to(device, dtype=torch.float)

    if load_existing_model:
        model.load_state_dict(
            torch.load(model_weights, map_location=device))


    optimizer = optimizer(model.parameters(), lr=learning_rate)

    imgs_train = ImageFolder(
        root=dataset_path.joinpath('train'),
        transform=DEF_PREPROCESSOR)
    dataloader_train = DataLoader(imgs_train, batch_size=batch_size, shuffle=True)

    imgs_valid = ImageFolder(
        root=dataset_path.joinpath('valid'),
        transform=DEF_PREPROCESSOR)
    dataloader_valid = DataLoader(imgs_valid, batch_size=batch_size, shuffle=True)

    train_length = sum([len(i) for i, _ in dataloader_train])
    logger.info(f'Length of train dataset: {train_length}')


    for i in range(epochs_count):
        es_time = time.time()

        counter = 0
        total_loss = 0

        for batch, ground_truth in dataloader_train:
            bs_time = time.time()

            batch = batch.to(device)

            ground_truth = torch.tensor(ground_truth)
            ground_truth = F.one_hot(
                ground_truth, num_classes=COUNT_OF_CANCER_TYPES)
            ground_truth = ground_truth.to(torch.float32)
            ground_truth = ground_truth.to(device)

            model.train()
            prediction = model(batch)
            prediction_tensor = prediction.logits

            # Compute the loss
            loss = criterion(prediction_tensor, ground_truth)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # Track total loss
            total_loss += loss.item()

            be_time = time.time()
            dur = be_time - bs_time
            logger.info(f'Batch: {counter} processing duration: {dur:.1f} sec')
            counter += 1

         # Calculate average loss for the epoch
        avg_loss = total_loss / train_length

        ee_time = time.time()
        dur = int(ee_time - es_time)
        logger.info(f'Epoch: {i}, total loss: {total_loss:.3f}, avg loss: {avg_loss:.3f}, duration: {dur} sec')
        if i % save_every_epochs == 0:
            inter_path = Path(str(model_weights) + f'_epoch_{i}')
            torch.save(model.state_dict(), inter_path)

        if i % evaluate_epochs_period == 0:
            avg_loss_train, accuracy_train, acc_cat_train = evaluate_accuracy(model, dataloader_train)
            avg_loss_valid, accuracy_valid, acc_cat_valid = evaluate_accuracy(model, dataloader_valid)
            logger.info(f'Epoch acc: {i}')
            logger.info(f'avg loss train {avg_loss_train:.3f}, acc train {accuracy_train:.3f}%')
            logger.info(f'avg loss valid {avg_loss_valid:.3f}, acc valid {accuracy_valid:.3f}%')

            for cat in acc_cat_train:
                cat_count = acc_cat_train[cat]['total']
                cat_acc = 100 * acc_cat_train[cat]['correct'] / acc_cat_train[cat]['total']
                logger.info(f'Train; cat: {cat} acc: {cat_acc:.3f}%, total count: {cat_count}')

            for cat in acc_cat_valid:
                cat_count = acc_cat_valid[cat]['total']
                cat_acc = 100 * acc_cat_valid[cat]['correct'] / acc_cat_valid[cat]['total']
                logger.info(f'Valid; cat: {cat} acc: {cat_acc:.3f}%, total count: {cat_count}')

    torch.save(model.state_dict(), model_weights)

if __name__ == "__main__":
    main()
