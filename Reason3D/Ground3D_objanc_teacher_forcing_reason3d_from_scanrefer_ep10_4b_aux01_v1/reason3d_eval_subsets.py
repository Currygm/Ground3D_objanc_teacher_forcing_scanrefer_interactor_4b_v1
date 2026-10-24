"""Reason3D source partitions and aggregation without additional inference."""
from torch.utils.data import ConcatDataset, Subset

SOURCES=("scannet","mp3d")
MODE_KEYS={"teacher":"teacher_answer","generated":"generated_answer","generated_no_fallback":"generated_no_fallback"}


def _source_labels(dataset):
    if isinstance(dataset, Subset):
        parent=_source_labels(dataset.dataset)
        return [parent[int(index)] for index in dataset.indices]
    if isinstance(dataset, ConcatDataset):
        return [source for child in dataset.datasets for source in _source_labels(child)]
    annotations=getattr(dataset,"annotations",None)
    if annotations is None or len(annotations)!=len(dataset):
        raise ValueError("Reason3D source grouping requires aligned annotation metadata")
    labels=[row.get("type") for row in annotations]
    if any(source not in SOURCES for source in labels):
        raise ValueError("Unknown Reason3D annotation source; expected scannet or mp3d")
    return labels


def source_indices(dataset):
    groups={source:[] for source in SOURCES}
    for index,source in enumerate(_source_labels(dataset)):
        groups[source].append(index)
    return groups


def merge_source_payloads(gathered,modes,source_counts,merge_standard,merge_no_fallback,examples_limit):
    overall,examples={},{}
    by_dataset={source:{"validation_samples":count} for source,count in source_counts.items()}
    for mode in modes:
        merge=merge_no_fallback if mode=="generated_no_fallback" else merge_standard
        all_payloads=[rank[mode][source] for rank in gathered for source in source_counts]
        overall[mode],examples[mode]=merge(all_payloads,examples_limit)
        if overall[mode]["samples"]!=sum(source_counts.values()):
            raise ValueError(f"Overall {mode} coverage differs from the validation dataset")
        for source,count in source_counts.items():
            metrics,_=merge([rank[mode][source] for rank in gathered],examples_limit)
            if metrics["samples"]!=count:
                raise ValueError(f"{source} {mode} coverage differs: {metrics['samples']} != {count}")
            by_dataset[source][MODE_KEYS[mode]]=metrics
    if "teacher" in modes and "generated" in modes:
        for stats in by_dataset.values():
            stats["selected_generated_gap"]=stats["teacher_answer"].get("obj/selected_iou",0)-stats["generated_answer"].get("obj/selected_iou",0)
    return overall,examples,by_dataset
