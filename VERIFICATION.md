# Release verification — 2026-11-23

## Completed

- Restored original DIOR source and the separately repaired VisDrone source from the live server into an isolated release tree.
- Retained upstream source licenses and rebuilt the Detrex config resource layout as ordinary files, making it ZIP-portable without depending on server symlinks.
- Parsed the three new Python commands and exercised their command-line help.
- Instantiated DIOR Baseline (58,056,040 parameters) and Full (59,458,983 parameters) on CPU with 20 classes and total batch 4.
- Instantiated VisDrone Baseline (58,035,490 parameters) and Full (59,437,793 parameters) on CPU with ten classes and total batch 2.
- Strictly loaded the complete archived EMA states for VisDrone Baseline 110k and Full 100k into these release models on CPU.
- Uploaded both complete VisDrone checkpoints and the common zero-update initialization to ModelScope. Remote file sizes and SHA-256 values matched local source files.
- Scanned release source for the provided account tokens/passwords and private-key markers. No matches were included.

CPU model checks used the restored Python 3.11 / PyTorch 2.3.1 environment and its existing compiled CUDA extensions, temporarily linked into an isolated audit directory. These binary links are not shipped. They validate source/model/checkpoint compatibility, rather than a fresh CUDA build on the destination machine.

## Pending hardware verification

The server returned `No devices were found` on 2026-11-23. No GPU training step, GPU inference, full AP reevaluation or clean CUDA extension build was performed for this release. The README scores are historical verified experiment results, not measurements from the new portable runner. Fresh installation and GPU smoke validation remain required before claiming fully verified end-to-end reproduction.

DIOR original best checkpoints remain in the existing ModelScope archive. Their recorded hashes are checked by the downloader. They were not downloaded again or strictly loaded during this release CPU audit.

The release packaging does not stop, start or change the existing training jobs. No historical training outputs or weights are deleted.
