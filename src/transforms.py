import albumentations as A

transform_album = A.Compose(
    [
        A.Rotate(limit=45),
        A.HorizontalFlip(),
        A.Blur(),
        A.Cutout(p=0.3, num_holes=16, max_h_size=16, max_w_size=16),
    ]
)
