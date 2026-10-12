# 0002: Speech model defaults

Date: 2026-10-12
Status: accepted

## Context

Spec 6.4 asks for a real-time factor of 0.5 or less for two streams on 8 CPU
cores without a GPU. A benchmark on 2026-10-12 decoded the five AMI clips (172 s
of speech in 300 s of audio) with 6 CPU threads (the `physical_cores - 2` rule)
on an AMD Ryzen 9 3900X:

| Model | Beam 1 | Beam 3 | Beam 5 | Latency for each utterance at beam 5 (p50 / p95 / max) |
|---|---|---|---|---|
| base.en | 0.19 | 0.17 | 0.19 | 0.55 / 0.71 / 0.81 s |
| small.en | 0.51 | 0.51 | 0.54 | 1.53 / 2.02 / 3.29 s |

The values are seconds of compute for each second of speech. Beam size does not
change the cost, because the encoder takes most of the time.

In a meeting, the two streams carry about 0.9 s of speech per second, and
partial decodes add about 1.5 times the cost of the finals. The estimates are
about 0.43 for base.en and about 1.2 for small.en. With small.en the transcript
falls behind on an 8-core machine. This machine has fast cores, so a typical
laptop is slower.

Word error rate on the AMI clips (mean of five): base.en 0.34, small.en 0.31.

## Decision

- Low and default tiers: `whisper-base-en`.
- High tier: `whisper-small-en`.
- `whisper-medium-en` leaves the manifest.
- Final beam size stays 5 (setting `asr_final_beam_size`).

## Consequences

- Live partials stay on for all tiers.
- Quality on hard meeting audio is about 3 points of word error rate lower than
  small.en. Users with 32 GB or more can choose the high tier.
- `forenote-cli bench asr` measures the cost on the user's machine.
