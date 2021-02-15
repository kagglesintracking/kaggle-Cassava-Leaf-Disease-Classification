import albumentations

def get_train_transforms(image_size):
    return albumentations.Compose([
        albumentations.RandomResizedCrop(image_size, image_size, p=1),
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ShiftScaleRotate(p=0.5),
        # albumentations.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        # albumentations.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.5),
        # albumentations.Resize(image_size, image_size),
        albumentations.Cutout(max_h_size=int(image_size * 0.3), max_w_size=int(image_size * 0.3), num_holes=1, p=0.5),
        # albumentations.CoarseDropout(p=0.5),
        albumentations.Normalize(),
    ])

def get_valid_transforms(image_size):
    return albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])