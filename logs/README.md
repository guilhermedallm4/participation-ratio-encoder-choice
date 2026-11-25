# Run logs

Verbatim stdout captures of the runs that produced the files in `results/`,
kept for traceability.

**These logs are in Portuguese.** They are the raw output of the scripts as
they existed when the experiments were run, before this repository was
translated to English. Rewriting them would make them a reconstruction rather
than a record, so they are left exactly as captured. Re-running the current
scripts produces the equivalent output in English.

| file | run |
|---|---|
| `pipeline.log` | the full pipeline |
| `download.log` | building the image samples |
| `measures.log`, `measures2.log` | geometry: participation ratio, field, arrangement |
| `probe.log`, `probe2.log` | linear probe |
| `batch2.log` | second batch of encoders |
| `field_shuffle.log` | receptive field and patch shuffling |
| `redo_flow.log` | residual flow, after the channels-last fix |
| `alignment2.log` | grid alignment control |
| `analysis.log` | final analysis |
| `log_stability.txt` | participation ratio against the number of images |
| `log_second.txt` | the held-out encoder run |
