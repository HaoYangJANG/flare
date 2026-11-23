"""Isolated diagnostic entry: confirm a possibly transient fork-before-exec PID.

The original frozen runner remains unchanged. A persistent second Python still
fails; memory/model/trace gates and all production functions are untouched.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools import diagnose_bin_evidence as runner


def confirm_processes(query, own_pid, clock=time.monotonic, pause=time.sleep, on_suspect=None):
    started, suspects = clock(), []
    while True:
        current = query()
        timestamp = clock()
        try:
            runner.validate_process_listing(current, own_pid)
        except RuntimeError:
            event = dict(elapsed_seconds=timestamp-started, listing=current)
            suspects.append(event)
            if on_suspect is not None:
                on_suspect(event)
            remaining = 1.0-(timestamp-started)
            if remaining <= 0:
                raise
            # No model work during confirmation; the deadline never resets.
            pause(min(.1, remaining))
        else:
            if not suspects:
                return None
            return dict(transient_python_seen=True, first=suspects[0]['listing'],
                        second=current, suspect_snapshots=suspects)


def main():
    from tools import evaluate_detail_floor as evaluator
    output = runner.validate_output(Path(sys.argv[sys.argv.index('--output')+1]))
    journal = output.parent/(output.name+'_process_confirmation.jsonl')
    source_manifest, process_check = evaluator._source_manifest, runner.check_processes
    def pinned_manifest():
        return {**source_manifest(), str(Path(__file__).resolve()): runner.sha256(__file__)}
    with journal.open('x') as stream:
        def checked():
            def record(event):
                stream.write(json.dumps(dict(own_pid=os.getpid(), **event))+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            def query():
                return subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True, timeout=1)
            evidence = confirm_processes(query, os.getpid(), on_suspect=record)
            if evidence is not None:
                record(evidence)
        evaluator._source_manifest, runner.check_processes = pinned_manifest, checked
        try:
            runner.main()
        finally:
            evaluator._source_manifest, runner.check_processes = source_manifest, process_check


if __name__ == '__main__':
    main()
