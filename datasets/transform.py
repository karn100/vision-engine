import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(config):

    aug_list = []

    if config["data"]["augment"]:
        aug_config = config["augmentations"]

        if aug_config.get("horizontal_flip",False):
            aug_list.append(A.HorizontalFlip(p=aug_config["horizontal_flip_p"]))
        
        if aug_config.get("random_crop",False):
            aug_list.extend([
                A.RandomCrop(
                    width=aug_config["crop_size"],
                    height=aug_config["crop_size"],
                    p=aug_config["random_crop_p"]
                )
            ])
           
        if aug_config.get("color_jitter",False):
            aug_list.append(A.ColorJitter(
                brightness=aug_config["brightness"],
                contrast=aug_config["contrast"],
                saturation=aug_config["saturation"],
                hue=aug_config["hue"],
                p=aug_config["color_jitter_p"]
            ))
        if aug_config.get("cutout",False):
            aug_list.append(A.CoarseDropout(
                max_holes=aug_config["cutout_holes"],
                max_height=aug_config["cutout_size"],
                max_width=aug_config["cutout_size"],
                fill=aug_config["cutout_fill"],
                p=aug_config["cutout_p"]
            ))
        
    aug_list.extend([
        A.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010)),
                    ToTensorV2()
    ])

    return A.Compose(aug_list)

def get_val_transform():

    return A.Compose([
        A.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)),
            ToTensorV2()
    ])
