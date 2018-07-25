import argparse
import os
import sys

from PIL import Image

from keras_retinanet.preprocessing.csv_generator import CSVGenerator

from keras_retinanet.utils.transform import random_transform_generator


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('images_dir', help='Path to train images folder.')
    parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    return parser.parse_args(args)

def main():
    args = parse_args(sys.argv[1:])

    transform_generator = random_transform_generator(
        min_rotation=-0.1,
        max_rotation=0.1,
        min_translation=(-0.1, -0.1),
        max_translation=(0.1, 0.1),
        min_shear=-0.1,
        max_shear=0.1,
        min_scaling=(0.9, 0.9),
        max_scaling=(1.1, 1.1),
    )

    generator = CSVGenerator(
        args.annotations,
        args.classes,
        # transform_generator=transform_generator,
        base_dir=args.images_dir
    )

    os.makedirs('gen_images', exist_ok=True)
    for i in range(5):
        batch = next(generator)
        img_ar = batch[0].reshape(batch[0].shape[1:]).astype('uint8')
        img = Image.fromarray(img_ar)
        img.save('gen_images/{}.jpg'.format(i+1))


if __name__=='__main__':
    main()