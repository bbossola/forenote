# 0003: LLM runtime

Date: 2026-10-12
Status: accepted

## Context

Spec B section 6.3 asks five questions before other code uses the LLM backend:

1. Does `llama-cpp-python` build on the baseline and load Gemma 4 E2B and E4B?
2. What are the prompt speed and the generation speed on CPU?
3. Does the prefix reuse work?
4. How much memory does the KV cache use at 8k, 16k and 32k?
5. Does the JSON-schema grammar work with Gemma 4?

The spike script is `spikes/llm/run_llm.py`. It loads a model and sends three
tasks with the same transcript prefix: a summary (500 tokens or less), a bullet
expansion (200 tokens or less) and action items with a JSON-schema grammar (500
tokens or less). The transcript is AMI meeting `ES2010c` (30.5 minutes). The
machine is an AMD Ryzen 9 3900X (12 cores, AVX2, no AVX512) with 62 GB RAM and
no GPU.

## Answers

1. Yes. `uv add "llama-cpp-python==0.3.36"` built the package in 41 s on this
   machine with CMake 3.28 and g++ (24 parallel jobs). The build uses
   `-march=native`. `llama_print_system_info()` reports
   `SSE3 SSSE3 AVX AVX2 F16C FMA BMI2 LLAMAFILE OPENMP REPACK`. The build shows
   only CMake developer warnings ("PUBLIC_HEADER files but no PUBLIC_HEADER
   DESTINATION") and a ccache note. Both models load. The architecture is
   `gemma4`.
2. See the table. E4B processes the prompt at 42 to 59 tokens/s and generates
   at 6 to 8 tokens/s.
3. Yes. The second task with the same prefix gets its first token 74 to 250
   times faster than the first task.
4. About 76 MiB for each 1,000 tokens of context with the default options.
   See "KV cache".
5. Yes. `json_ok` is true in all runs. The grammar lowers the generation speed
   by about 25 % (6.0 to 4.6 tokens/s for E4B).

In all runs, the generation stops at the end of the model turn. No output
contains chat template text.

## Measurements

All runs use `n_batch=512` and the default options of `Llama` (no flash
attention, full-size SWA cache). RSS is the growth of the process RSS after the
load and includes the mapped model file. "First token" is the time from the
call to the first generated token. Generation speed is from the first task.

| Model | Context | Threads | Load | RSS after load | Prompt tokens | Prompt speed | First token, task 1 | First token, task 2 | Generation speed |
|---|---|---|---|---|---|---|---|---|---|
| E2B (10 min) | 8,192 | 7 | 2.0 s | 4,479 MiB | 2,635 | 105.1 tok/s | 25.07 s | 0.34 s | 15.0 tok/s |
| E4B (30 min) | 16,384 | 7 | 3.8 s | 8,344 MiB | 9,984 | 42.7 tok/s | 233.86 s | 0.94 s | 6.0 tok/s |
| E4B (30 min) | 16,384 | 11 | 3.5 s | 8,344 MiB | 9,984 | 59.1 tok/s | 168.99 s | 0.68 s | 6.2 tok/s |
| E4B (30 min) | 32,768 | 7 | 4.1 s | 9,560 MiB | 9,984 | 41.9 tok/s | 238.38 s | 0.95 s | 6.0 tok/s |
| E4B (10 min) | 8,192 | 7 | 3.2 s | 7,736 MiB | 2,635 | 54.9 tok/s | 48.01 s | 0.62 s | 8.1 tok/s |

The 30-minute transcript is 9,984 tokens. It fits in 16k but not in 8k.

Two extra runs (E4B, 16k, 7 threads, 30 minutes) check two options that
llama.cpp reports on stderr ("FA is not enabled", "using full-size SWA cache"):

| Options | RSS after load | Prompt speed | First token, task 1 | First token, task 2 | Generation speed |
|---|---|---|---|---|---|
| default | 8,344 MiB | 42.7 tok/s | 233.86 s | 0.94 s | 6.0 tok/s |
| `flash_attn=True` | 8,024 MiB | 51.9 tok/s | 192.28 s | 1.63 s | 6.9 tok/s |
| `flash_attn=True, swa_full=False` | 7,424 MiB | 52.3 tok/s | 190.79 s | 1.61 s | 7.4 tok/s |

With `swa_full=False` the prefix reuse still works. The bullet and JSON outputs
are the same text as in the default run.

### KV cache

E4B with the default options: RSS after load is 7,736 MiB at 8k, 8,344 MiB at
16k and 9,560 MiB at 32k. The KV cache grows by 608 MiB from 8k to 16k and by
1,216 MiB from 16k to 32k. That is about 76 MiB for each 1,000 tokens, so the
KV cache is about 1.2 GB at 16k and 2.4 GB at 32k. The model weights and the
compute buffers use the remaining 7.1 GB.

### Output check (E4B, 16k)

- The summary describes the conceptual design meeting for a remote control. It
  gives the presentations, the retro telephone dial idea, the latex cover, the
  choice of batteries and the yellow button light. All facts are in the
  transcript.
- The bullet expansion of "remote control buttons" uses transcript facts: large
  labelled buttons, buttons that light on touch, the dial layout, and yellow
  light against blue buttons.
- The action items have two items. The owners are `I_D_` and `U_I_D_`. These
  are the roles (industrial designer, user interface designer) as the AMI
  transcript spells them. The texts are short phrases from the transcript.
  E4B at 8k (10 minutes) gives one item with owner `null`.

## Estimate for the 30-minute notes job

The formula is: prefix time of task 1 + 6 bullets × (first token of a repeated
task + 100 tokens ÷ generation speed) + summary (300 tokens) + action items
(200 tokens).

| Run (E4B, 16k) | Prefix | 6 bullets | Summary | Action items | Total |
|---|---|---|---|---|---|
| 7 threads, default | 233.9 s | 105.7 s | 50.0 s | 33.3 s | 423 s (7.0 min) |
| 11 threads, default | 169.0 s | 100.9 s | 48.4 s | 32.3 s | 351 s (5.8 min) |
| 7 threads, `flash_attn`, `swa_full=False` | 190.8 s | 90.7 s | 40.5 s | 27.0 s | 349 s (5.8 min) |

The target of spec 6.4 is 3 minutes or less on 8 cores. The default tier misses
it by a factor of 2 or more. The prefix alone takes more than 3 minutes with 7
threads. This machine has fast cores, so a typical 8-core laptop is slower. The
grammar makes the action items about 25 % slower than the formula shows.

## Decision

- Backend: `llama-cpp-python==0.3.36`, built from source for the CPU.
- Options: `flash_attn=True` and `swa_full=False`. Together they make the
  prompt 22 % faster and the generation 23 % faster, and they use 920 MiB less
  memory at 16k. The prefix reuse works with them.
- Context size: keep 8k (low), 16k (default) and 32k (high). A 30-minute
  transcript is about 10,000 tokens, so 16k is the smallest size that holds it.
  With the options above, E4B at 16k uses 7.4 GB after load and 7.7 GB after
  three tasks. That is at the top of the 7–8 GB of spec 9.1.
- Threads: LLM jobs use `n_threads = n_threads_batch = physical_cores - 1`.
  The notes job runs after the meeting, when ASR is idle. On this machine, 11
  threads make the prompt 38 % faster than 7 threads. The generation speed does
  not change, because memory bandwidth limits it.
- Prompt rendering: read the chat template from the GGUF metadata
  (`tokenizer.chat_template`). Render it with
  `llama_cpp.llama_chat_format.Jinja2ChatFormatter(template=..., eos_token=...,
  bos_token=...)`. Get the token texts with
  `llm._model.token_get_text(llm.token_bos())` and
  `llm._model.token_get_text(llm.token_eos())` (`<bos>` and `<eos>`).
  Tokenize with `llm.tokenize(prompt.encode("utf-8"), add_bos=False,
  special=True)`, because the rendered prompt already starts with `<bos>`.
- No stop strings are necessary. llama.cpp stops at the Gemma 4 end-of-turn
  token.

### API facts (llama-cpp-python 0.3.36)

- The Gemma 4 template is `<bos><|turn>user\n{text}<turn|>\n<|turn>model\n`. It
  does not use the Gemma 3 marks `<start_of_turn>` and `<end_of_turn>`.
- `Jinja2ChatFormatter.__init__(template, eos_token, bos_token,
  add_generation_prompt=True, stop_token_ids=None)`. Calling it with
  `messages=[...]` returns an object with `.prompt`.
- `Llama.tokenize(text: bytes, add_bos=True, special=False)`. The first
  parameter is `text`, not `data`.
- `LlamaGrammar.from_json_schema(json_schema: str, verbose=True)`.
- `Llama(...)` accepts `flash_attn` (default `False`) and `swa_full` (default
  `None`, which keeps the llama.cpp default of a full-size SWA cache).
- `token_eot()` returns -1 for these models. The end-of-turn token is an
  end-of-generation token in the vocabulary, and the generation stops there.

## Consequences

- The README lists CMake and a C++ compiler as build requirements.
- The speed target of spec 6.4 does not hold with E4B on CPU. The project owner
  must decide before plan B2. The options are a longer target, E2B for the
  default tier (about two times faster at 2,600 tokens: 105 against 55 tok/s
  prompt, 15 against 8 tok/s generation), or a shorter prompt.
- Memory at 16k is close to the limit of spec 9.1. Use the options above.
- `swa_full=False` keeps only the sliding window for the SWA layers. The spike
  shows that the prefix reuse works when only the tail of the prompt changes.
  Plan B1 Task 4 must keep a test with two tasks on the same prefix.
- `forenote-cli bench llm` (plan B2) must measure the prompt speed and the
  generation speed on the user's machine.
