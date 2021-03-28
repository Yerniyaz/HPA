import albumentations as A
from transforms import transform_album


backbone = 'efficientnet-b4'  # efficientnet-b0, se_resnext50_32x4d, mobilenet etc.

img_w = 512
img_h = 512

img_ext = '.png'
checkpoint_ext = '.pth'

train_key = 'train'
val_key = 'val'


cell_types = ['Nucleoplasm', 'Nuclear membrane', 'Nucleoli', 'Nucleoli fibrillar center', 'Nuclear speckles',
              'Nuclear bodies',	'Endoplasmic reticulum', 'Golgi apparatus', 'Intermediate filaments', 'Actin filaments',
              'Microtubules', 'Mitotic spindle', 'Centrosome', 'Plasma membrane', 'Mitochondria', 'Aggresome',
              'Cytosol', 'Vesicles and punctate cytosolic patterns', 'Negative']

train_csv = ['all_images.csv']

val_csv = ['all_images.csv']

params = {
    'train': {
        'transform': transform_album,
        'train': True,
        'csv_files': train_csv
    },
    'val': {
        'transform': A.Compose([]),
        'train': False,
        'csv_files': val_csv
    }
}
