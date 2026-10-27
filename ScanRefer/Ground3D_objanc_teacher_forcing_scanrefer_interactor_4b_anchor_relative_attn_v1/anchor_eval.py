"""ScanRefer subset metadata and prediction-only evaluation audit records."""
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path

NYU18={'cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','desk','curtain','refrigerator','shower curtain','toilet','sink','bathtub','garbage bin'}


def unique_multiple_lookup(raw_rows,label_rows):
    raw2class={r['raw_category']:r['nyu40class'] if r['nyu40class'] in NYU18 else 'others' for r in label_rows}
    scenes=defaultdict(dict)
    for row in raw_rows:
        name=row['object_name'].replace('_',' ')
        scenes[str(row['scene_id'])][str(row['object_id'])]=raw2class.get(name,'others')
    out={}
    for scene,objects in scenes.items():
        counts=Counter(objects.values())
        for oid,category in objects.items():out[(scene,oid)]='unique' if counts[category]==1 else 'multiple'
    return out


def load_subset_lookup(config):
    root=Path(config['data']['scene_root']).parent
    raw=root/'ScanRefer/ScanRefer_filtered_val.json'
    labels=root/'scannetv2/scannetv2-labels.combined.tsv'
    with labels.open() as h:rows=list(csv.DictReader(h,delimiter='\t'))
    return unique_multiple_lookup(json.loads(raw.read_text()),rows)


def validate_coverage(payloads,total):
    for mode in ('teacher','generated'):
        ids=[i for p in payloads for i in p[mode]['sample_indices']]
        keys=[i for p in payloads for i in p[mode]['sample_ids']]
        if sorted(ids)!=list(range(total)) or len(set(keys))!=total:raise ValueError('Missing or duplicate evaluation IDs')
