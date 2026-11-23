"""Build the archived DIOR COCO layout from existing images and JSON annotations."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--copy', action='store_true', help='Copy instead of creating image symlinks')
    args = p.parse_args()
    index = {}
    for image in args.images.rglob('*'):
        if image.is_file() and image.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            index.setdefault(image.name, []).append(image.resolve())
    for split in ['train', 'val']:
        annotation = args.annotations / f'instances_{split}.json'
        data = json.loads(annotation.read_text())
        if len(data['categories']) != 20:
            raise ValueError('DIOR requires 20 categories')
        target = args.output / 'images' / split
        target.mkdir(parents=True, exist_ok=True)
        for row in data['images']:
            rel = Path(row['file_name'])
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Unsafe image filename in annotation')
            candidates = index.get(rel.name, [])
            if len(candidates) != 1:
                raise ValueError(f'Missing or ambiguous image: {rel.name}')
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copyfile(candidates[0], dst) if args.copy else dst.symlink_to(candidates[0])
        folder = args.output / 'annotations'
        folder.mkdir(exist_ok=True)
        shutil.copyfile(annotation, folder / annotation.name)
        print(split, len(data['images']), len(data['annotations']))


if __name__ == '__main__':
    main()
