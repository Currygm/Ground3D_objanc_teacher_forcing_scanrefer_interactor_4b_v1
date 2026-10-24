import json
import threading
import time
import pytest
from ground3d_migration.eval_shards import publish_shard, wait_for_shards


def test_four_shards_finish_independently_and_rank_zero_waits(tmp_path):
    run_id = 'fresh-run'
    directory = tmp_path / run_id
    directory.mkdir()
    finished = []
    def worker(rank, delay):
        time.sleep(delay)
        publish_shard(directory, run_id, rank, 4, {'samples': list(range(rank, 11, 4))})
        finished.append(rank)
    workers = [threading.Thread(target=worker, args=(r, .05 if r != 3 else .6)) for r in range(4)]
    for worker_thread in workers: worker_thread.start()
    time.sleep(.2)
    assert set(finished) == {0, 1, 2}  # early workers do not wait for slower rank
    reports = []
    payloads = wait_for_shards(directory, run_id, 4, timeout=3, poll_interval=.02,
                              report_interval=.1, report=reports.append)
    for worker_thread in workers: worker_thread.join()
    assert sorted(i for payload in payloads for i in payload['samples']) == list(range(11))
    assert reports and '3' in reports[0]
    assert not list(directory.glob('*.tmp'))


def test_timeout_identifies_missing_ranks(tmp_path):
    publish_shard(tmp_path, 'run', 0, 4, {'ok': True})
    with pytest.raises(TimeoutError, match='1, 2, 3'):
        wait_for_shards(tmp_path, 'run', 4, timeout=.1, poll_interval=.01, report=lambda x: None)


def test_rejects_stale_wrong_run_metadata(tmp_path):
    publish_shard(tmp_path, 'stale', 0, 1, {})
    with pytest.raises(ValueError, match='metadata'):
        wait_for_shards(tmp_path, 'new', 1, timeout=1, report=lambda x: None)


def test_invalid_timeout_and_failed_publish_have_no_complete_file(tmp_path):
    with pytest.raises(ValueError):
        wait_for_shards(tmp_path, 'run', 1, timeout=0)
    with pytest.raises(ValueError):
        publish_shard(tmp_path, 'run', 0, 1, {'bad': float('nan')})
    assert not (tmp_path / 'rank_00000.json').exists()


def test_evaluator_does_not_use_end_of_eval_collectives():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / 'evaluate_segmentation.py').read_text()
    assert 'dist.gather_object' not in source
    evaluation = source[source.index('local_payload = {'):]
    assert 'dist.barrier' not in evaluation
    assert 'wait_for_shards' in evaluation and 'publish_shard' in evaluation


def test_four_real_processes_early_workers_exit_before_slow_rank(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    code = """import json,sys,time
from pathlib import Path
from ground3d_migration.eval_shards import publish_shard,wait_for_shards
rank=int(sys.argv[1]);directory=Path(sys.argv[2])
if rank==3:
    while not (directory/'release_slow').exists():time.sleep(.02)
publish_shard(directory,'process-run',rank,4,{'indices':list(range(rank,13,4))})
if rank==0:
    rows=wait_for_shards(directory,'process-run',4,timeout=30,poll_interval=.02,report=lambda x:None)
    (directory/'merged.json').write_text(json.dumps(rows))
"""
    children = [subprocess.Popen([sys.executable, '-c', code, str(r), str(tmp_path)], cwd=root)
                for r in range(4)]
    try:
        assert children[1].wait(timeout=20)==0
        assert children[2].wait(timeout=20)==0
        assert children[0].poll() is None and children[3].poll() is None
        (tmp_path/'release_slow').touch()
        assert all(child.wait(timeout=20)==0 for child in children)
        rows=json.loads((tmp_path/'merged.json').read_text())
        assert sorted(i for row in rows for i in row['indices'])==list(range(13))
    finally:
        for child in children:
            if child.poll() is None:child.kill()
        for child in children:child.wait()
