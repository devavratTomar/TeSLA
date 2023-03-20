from .kather2016 import get_kather2016_datasets
from .kather100k import get_kather100k_datasets
from .config import split


get_dataset_funcs = {
    'kather2016': get_kather2016_datasets,
    'kather100k': get_kather100k_datasets,
}

def get_class_splits(dataset):

    if dataset == 'kather2016':

        known_classes = split[dataset]
        open_set_classes = [x for x in range(8) if x not in known_classes]
        print('training on known classes:', known_classes)

    elif dataset == 'kather100k': # this one has nine classes
        known_classes = split[dataset]
        open_set_classes = [x for x in range(9) if x not in known_classes]
        print('training on known classes:', known_classes)

    else:

        raise NotImplementedError

    return known_classes, open_set_classes
