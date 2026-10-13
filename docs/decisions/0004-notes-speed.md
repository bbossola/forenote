# 0004: Notes speed

Date: 2026-10-13
Status: accepted

## Context

Spec B section 6.4 sets two targets for the default tier on 8 cores. The notes
must be ready 180 s or less after the stop of a 30-minute meeting. Speech
recognition must run at 0.5 s or less for each second of speech while the live
reading runs.

Decision 0003 estimated 160 s for the notes with live reading (E4B, 16k,
`flash_attn`, `swa_full=False`). Plan B3 built the live reading. This decision
records the measurement with `forenote-cli bench llm`.

The machine is an AMD Ryzen 7 5825U (8 physical cores, 16 threads, AVX2) with 62
GB RAM and no GPU. It is a laptop CPU. It is not the machine of decision 0003
(Ryzen 9 3900X, 12 cores). The model is `gemma-4-e4b-q4_0` with a context of
16,384 tokens. The fixture is the 30-minute AMI transcript (10,197 tokens). The
speech cost uses the clips of `tests/fixtures/ami/`.

The task brief says that the machine has 16 physical cores. That is wrong. It
has 16 hardware threads on 8 physical cores. Because of this, the default thread
count of the bench is already 7 (and 4 for the live reading). Run 4 is the same
configuration as run 1, and it measures the noise. Run 6 (15 threads) is an extra
run that uses all hardware threads.

The machine was not idle. A `clamscan` process used one core during all runs.
Other jobs (Java builds) used many cores during run 1 and some other runs. The
numbers below have noise. Runs 1 and 4 have the same thread and option values.

## Results

Full outputs are in the report of Task 10.

| Run | Threads | Live threads | Options | Prompt tok/s | Generation tok/s | Notes without live reading | Live steps (compute) | Notes from the stop | Speech cost | Memory (anonymous / file) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (target) | 7 | 4 | flash on, SWA full off, cold step, clips | 28.3 | 9.1 | 444 s | 59 (476 s) | 166 s | 0.31 | 3,285 / 4,978 MiB |
| 2 | 7 | 4 | flash off | 32.9 | 9.2 | not run | 59 (457 s) | 128 s | not run | 3,126 / 4,929 MiB |
| 3 | 7 | 4 | SWA full on | 27.9 | 9.2 | not run | 59 (417 s) | 145 s | not run | 3,525 / 4,929 MiB |
| 4 | 7 (default) | 4 (default) | defaults | 28.5 | 9.2 | not run | 59 (388 s) | 138 s | not run | 2,902 / 4,929 MiB |
| 5 | 3 | 2 | defaults | 19.1 | 8.8 | not run | 59 (640 s) | 205 s | not run | 2,895 / 4,929 MiB |
| 6 (extra) | 15 | 8 | defaults | 24.2 | 2.6 | not run | 59 (344 s) | 303 s | not run | 3,165 / 4,929 MiB |

Runs 2 to 6 use `--no-cold` and no clips, so they have no notes-without-live
value and no speech cost.

## Target check

- Notes from the stop: 166 s in the target run. The target is 180 s. The
  target holds, with a margin of 14 s (8 %). The other 7-thread runs give 128 to
  145 s. The spread between runs 1 and 4 (166 s and 138 s, same values) shows
  that the noise is about 30 s on this busy machine.
- Speech cost: 0.31 s for each second of speech. The target is 0.5. The target
  holds.
- Notes without live reading: 444 s (7.4 min). This is 2.5 times the target and
  is slower than the 349 s of decision 0003. This machine has a slower CPU
  (prompt 28 tok/s against 52 tok/s). The live reading is the reason that the
  target holds.
- The live reading computes for 388 to 476 s over a recording of about 30
  minutes. The live reading keeps up with the recording.
- Limit of this evidence: one machine, one meeting, and a busy machine. The
  margin is small. A slower 8-core PC can miss the target.

## Options

Run 1 and run 4 (flash on, SWA full off) give a prompt speed of 28.3 and 28.5
tok/s. Run 2 (flash off) gives 32.9 tok/s, which is 16 % faster, and a stop time
of 128 s. Run 3 (SWA full on) gives 27.9 tok/s and 145 s, and it uses 240 to
620 MiB more anonymous memory. The generation speed is the same in all runs
(9.1 to 9.2 tok/s).

The stop time spread inside one configuration (138 s to 166 s) is as large as
the spread between the options. One run for each option cannot show a gain.
Decision: keep `flash_attn=True` and `swa_full=False`. They use less memory. Do
not change the default values from one noisy run. A quiet machine must measure
`--flash-attn off` again, because its prompt speed is higher in this run.

The first-token difference of decision 0003 (0.94 s against 1.61 s) is not
explained. The bench does not measure the first token of a repeated task.

## Thread rule

The rule is `physical_cores - 1` for the notes job and half of the physical
cores for the live reading. On this machine it gives 7 and 4.

| Threads | Prompt tok/s | Generation tok/s |
|---|---|---|
| 3 | 19.1 | 8.8 |
| 7 | 28.3 to 28.5 | 9.1 to 9.2 |
| 15 | 24.2 | 2.6 |

The rule is the best of the three values. Fewer threads are slower. All
hardware threads are slower than the rule, and the generation falls by a factor
of 3.5. This is more than the cost of two threads on a busy machine, so the
15-thread run is also a sign of oversubscription (the other jobs used cores at
that time). One machine cannot confirm the rule. The bench runs on other
machines, and the owner must collect the results.

## Memory

The bench reports 3.3 GB anonymous memory and 5.0 GB file-backed memory (the
mapped model file) for the LLM at 16k. The sum is 8.2 GB. Run 4 and run 5 give
2.9 GB anonymous. The anonymous part is the KV cache and the compute buffers.
The file-backed part can leave the RAM under pressure, and the OS reads it again
from the disk.

The budget of spec 9.1 is 7-8 GB for the LLM, Whisper and the embeddings
together. The LLM alone uses 8.2 GB when the file pages count. With Whisper and
the embeddings, the total is above the budget. The anonymous memory alone (3.3
GB) is inside the budget with room for Whisper. Nobody measured the total with
ASR and the live reading on a 16 GB machine. The memory item stays open (see the
carry note).

## Abort callback

The generation speed of run 1 is 9.1 tok/s. Decision 0003 measured 6 to 7 tok/s
on another CPU, with no callback. The speed here is not lower than the old
value, and all runs give 8.8 to 9.2 tok/s (except the oversubscribed run 6). The
token loop shows no cost from the abort callback. The two machines are different,
so this is a rough check.

## Quality

Task 7 tuned the prompts against the AMI quality fixtures (E2B). All 3 meetings
(ES2002a, IS1009a, TS3003a) pass with seeds 1, 2 and 3. No `xfail` stays. The
check uses only 3 seeds and the results vary between seeds. The evidence is
weak for the E4B default model, because the tests use E2B.

## Decision

- Keep the defaults: `flash_attn=True`, `swa_full=False`, context 16k, threads
  `physical_cores - 1`, live threads half of the physical cores. No default
  value changes in this task.
- The speed targets of spec 6.4 hold on this machine: 166 s against 180 s, and
  a speech cost of 0.31 against 0.5. The margin of the notes time is small.
- The default tier needs the live reading to reach the target. Without it, the
  notes take 444 s.
- Open items go to the section "Next plans: from plan B3" of the carry note:
  options on a quiet machine, the thread rule on other machines, and the total
  memory.
