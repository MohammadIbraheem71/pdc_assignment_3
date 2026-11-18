# Part 5: MPI collectives and communication-cost reasoning

This folder contains the analytical Part 5 answers, the numerical verifier, and the
two diagrams. There is no MPI implementation and no compilation step.

## Files

| file | purpose |
|---|---|
| `part5_notes.md` | full written answers: A data flow, B derivations, C numbers, D topology steps, E load balance, assumptions and follow-ups |
| `verify_part5.py` | recomputes every number, asserts expected values, prints the report and writes `part5_output.txt` |
| `part5_output.txt` | captured output of the verifier (source of truth for all numbers in the notes) |
| `draw_part5_diagrams.py` | draws both diagrams with Pillow |
| `part5_dataflow.png` | MPI data-flow diagram (broadcast -> local tiles -> Gatherv -> reassemble) |
| `part5_topologies.png` | p=16 ring, 4x4 mesh and Q4 hypercube with broadcast steps marked |

## Requirements

- Python 3 (tested with the system Python 3).
- Pillow (`PIL`) for the diagrams, version 12.1.1 on this machine.
- `matplotlib` and graphviz `dot` are **not** installed here, so the diagrams use
  Pillow with explicit geometry. This is also recorded in `plan.md`.

## Regenerate the numbers

Run from the repository root:

```bash
python3 part5/verify_part5.py
```

The script prints the whole report and also writes it to `part5/part5_output.txt`.
The last lines should be:

```text
All checks passed
Every printed number was recomputed and matched the expected value.
```

It performs 57 checks covering the data sizes, the Scatterv/Gatherv example, the
broadcast derivations, the `p=8, m=2^20` numbers, the topology step counts and the
load-balance/LPT simulation. A mismatch exits non-zero, so it is safe to use in a
quick check:

```bash
python3 part5/verify_part5.py && echo "Part 5 numbers verified"
```

## Regenerate the diagrams

Run from the repository root:

```bash
python3 part5/draw_part5_diagrams.py
```

This overwrites:

- `part5/part5_dataflow.png` (1640x1140);
- `part5/part5_topologies.png` (1900x1270).

Both are plain PNGs; open them with any image viewer. The topology diagram marks the
round in which each node receives the message (`sN`), the round on each ring edge, the
mesh wavefront, and the hypercube dimension fixed in each round.

## Read the answers

Open `part5/part5_notes.md`. Its section numbers match the assignment:

- **A** data-flow design, collectives, `counts`/`displs` example, non-contiguous
  tiles, full-circle broadcast recommendation, data sizes;
- **B** linear and binomial derivations, p=8 round table, large-message note;
- **C** every intermediate step for `m=2^20`, `alpha=10 us`, `beta=1 ns/B`;
- **D** ring / 4x4 mesh / Q4 step counts and the limits of step-count reasoning;
- **E** original makespan/speedup/efficiency/idle, the LPT split, and the
  communication-cost comparison.

## Notes on regeneration

- The verifier is the source of truth. If you change an assumption in the script,
  re-run it before changing the prose or diagrams.
- The diagrams do not contain measured data. They show the model assumptions and the
  analytical step counts.
- The task-E communication estimate uses an assumed 32x32 RGBA float32 tile
  (16,384 bytes) and a 16-byte request. Those assumptions are stated in the script
  and in section E.3 of the notes.
- No files outside `part5/` are needed to regenerate anything in this folder.
