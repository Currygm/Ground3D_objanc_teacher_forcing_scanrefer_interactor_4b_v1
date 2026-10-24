"""Download the pinned official initialization asset, or verify an offline copy."""
from pathlib import Path
import argparse,hashlib,json

REPO='CH3COOK/bert-base-embedding'
REVISION='dad6c5c4bbc67801c70d59f8dfd6446925f07cc1'
MODEL_SHA='2cf83467799040bad74bc71b81d3fe2bfbe06b166ca2d21b3cfc784124b27b18'
LL3DA_REVISION='ebbdd771b616ce3714932c3627d91ece6d8d6cd1'
FILES=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt']

def verify(root):
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        return h.hexdigest()
    files={name:digest(root/name) for name in FILES}
    if files['model.safetensors']!=MODEL_SHA:raise ValueError('Official initialization model SHA256 mismatch')
    return {'repo':REPO,'revision':REVISION,'official_ll3da_revision':LL3DA_REVISION,
            'files':files,'official_model_sha256':MODEL_SHA}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parent/'models/bert-base-embedding'
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(REPO,revision=REVISION,allow_patterns=FILES,local_dir=root)
    manifest=verify(root)
    (root/'asset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
