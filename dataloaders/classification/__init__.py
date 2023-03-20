from .cifar import CIFAR, CIFAR_C
from .visda import VisDA_C
from .imagenet import IMAGENET_C
from .kather.open_set_datasets import get_class_splits
from .kather.kather100k import get_kather100k_datasets
from .kather.kather2016 import get_kather2016_datasets

def get_dataset(dataset_name, root_dir, opt, load_source=False, aug_mult=0, hard_augment="randaugment"):

    if dataset_name == "visda":
        return VisDA_C(root_dir, "train" if load_source else "validation", aug_mult=aug_mult, hard_augment=hard_augment)
    elif "cifar" in dataset_name:
        if load_source:
            return CIFAR(root_dir, dataset_name, False)
        else:
            return CIFAR_C(root_dir, dataset_name, opt.corruption, opt.corruption_level, aug_mult=aug_mult)
    elif "imagenet" in dataset_name:
        return IMAGENET_C(root_dir, corruption=opt.corruption, level=opt.corruption_level, aug_mult=aug_mult)
    elif "kather" in dataset_name:
        if load_source:
            known_classes, _ = get_class_splits('kather100k')
            return get_kather100k_datasets(root_dir=root_dir, known_classes=known_classes, seed=opt.seed)
        else:
            known_classes, _ = get_class_splits('kather2016')
            return get_kather2016_datasets(root_dir=root_dir, known_classes=known_classes, seed=opt.seed, aug_mult=aug_mult)
    else:
        raise NotImplementedError(f"The dataset {dataset_name} is not implemented yet !")
