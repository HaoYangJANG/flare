"""Download verified project assets without embedding account credentials."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['dior', 'visdrone', 'initialization'], required=True)
    p.add_argument('--output', type=Path, default=Path('weights'))
    p.add_argument('--archive', type=Path, help='Existing DIOR archive, avoiding another download')
    args = p.parse_args()
    from modelscope_hub import HubApi
    api = HubApi(token=os.environ.get('MODELSCOPE_API_TOKEN'))
    repo = 'etheral11/diffudetr'
    args.output.mkdir(parents=True, exist_ok=True)
    if args.dataset != 'dior':
        if args.dataset == 'visdrone':
            manifest = api.download_file(repo, 'dataset', 'visdrone/weights_manifest.json')
            files = [x for x in json.loads(manifest.read_text()) if x['path'].startswith('visdrone/')]
        else:
            files = [dict(path='initialization/common_init_coco_derived_fresh_dior_seed42.pth',
                          sha256='fbb441c67ad7ae3a0a51033e64f6ef49e4e73b13e2ef4a43acaad77c6440abbe')]
        for item in files:
            source = api.download_file(repo, 'dataset', item['path'], expected_sha256=item['sha256'])
            destination = args.output / Path(item['path']).name
            shutil.copyfile(source, destination)
            if sha256(destination) != item['sha256']:
                raise ValueError('Checksum mismatch')
            print(destination)
        return
    archive = args.archive or api.download_file(repo, 'dataset', 'all_ablations_best_20261017.tar.zst',
        expected_sha256='96a102f4465f02a14003baae689e1a6267f6c0de31f6cab71d716c60e6b596b4')
    if sha256(archive) != '96a102f4465f02a14003baae689e1a6267f6c0de31f6cab71d716c60e6b596b4':
        raise ValueError('Wrong DIOR archive')
    import zstandard
    with archive.open('rb') as stream, zstandard.ZstdDecompressor().stream_reader(stream) as decoded:
        with tarfile.open(fileobj=decoded, mode='r|') as tar:
            for member in tar:
                name = member.name.lstrip('./')
                selected = None
                for arm in ['baseline', 'full']:
                    if name.startswith(f'experiments/{arm}/weights/') and name.endswith('.pth'):
                        selected = f'dior_{arm}_best.pth'
                if name.startswith('data/annotations/') and name.endswith('.json'):
                    selected = Path(name).name
                if selected and member.isfile():
                    with tar.extractfile(member) as src, (args.output / selected).open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                    expected = {
                        'dior_baseline_best.pth':'c833009205a050f14cdeaf68ba0d09c7a421b6cb348edb0ba0fd2b6aa114a24e',
                        'dior_full_best.pth':'a72386122a92185b4d25bb5418a22abac4b82f8a717cfe34fc1333c084641412',
                    }.get(selected)
                    if expected and sha256(args.output / selected) != expected:
                        raise ValueError('Extracted DIOR checkpoint checksum mismatch')
                    print(args.output / selected)


if __name__ == '__main__':
    main()
