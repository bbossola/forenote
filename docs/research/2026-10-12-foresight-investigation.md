# Google AI Edge Foresight: investigation for a Linux version

Date: 2026-10-12

## 1. Summary

Google AI Edge Foresight is a closed-source meeting notetaker for macOS on
Apple Silicon. It also does local RAG over the files of the user. All the AI
inference runs on the device. Google announced it on 2026-10-06.

A Linux version is possible. The UI technology, the inference runtime, the LLM,
and the embedding model are all available on Linux x86_64. The speech
recognition models of Foresight are Google-internal. A Linux version must use
an open model for speech recognition, for example Whisper.

We do not reuse code or models from Foresight. The analysis in this document
comes from the public product page, press reports, and the readable strings in
the public installer. We did not run, decompile, or modify the binaries.

## 2. Product page

Source: <https://developers.google.com/edge/foresight>. The page body is in an
iframe:
<https://developers.google.com/frame/edge/foresight/index_c3d0dc1863c7b1d97c990d60ddae913c65eddb7dc59f8f2fb319eb5898f587ae.frame>.

The page states these features:

- **Automated synthesis.** The app listens to the meeting. The user writes
  shorthand bullets. The app changes the bullets into full notes with the
  transcript. It works for video calls and for face-to-face meetings.
- **Local knowledge.** The user points the app to project folders, reference
  material, diagrams, and the calendar. Local folders and Google Drive are
  supported.
- **Meeting intelligence.** Chat over the knowledge of the user. "Live
  Assistance" listens and answers questions automatically, with context cards
  and citations.
- **Models.** EmbeddingGemma 2 (740M parameters, text, image, video, and audio
  in one vector space) and Gemma 4.
- **Formats.** PDF, Google Docs, Microsoft Office formats, plain text,
  Markdown, and web bookmarks.
- **Privacy.** Audio, transcripts, and files stay on the computer. The app
  works offline.
- **Hardware.** "Optimized for Apple Silicon". No other requirement is given.

Download: <https://dl.google.com/foresight-mac/foresight_mac.dmg>.

## 3. Installer analysis

### 3.1 Installer

| Item | Value |
|---|---|
| File | `foresight_mac.dmg`, 160,266,104 bytes |
| SHA-256 | `ba92290de9aee0e6287ea161a86bdfb71c8b052994492b4c5f76da854bee5d6d` |
| Last-Modified | 2026-10-06 03:04:39 GMT |
| Bundle ID | `com.google.AIEdgeForesight` |
| Version | 1.0.0, build `1.0.0.28.20261005` |
| Helper version | 1.0.3 |
| Minimum macOS | 14.2 |
| Architecture | Mach-O arm64 only |

Method: `7z x` on the DMG, `tar` on the helper archive, `strings -n 6` on the
two executables, then `grep`.

### 3.2 Bundle content

| Path | Size | Purpose |
|---|---|---|
| `MacOS/Foresight` | 150 MB | Flutter app (Dart AOT) with native plugins |
| `Resources/ForesightHelper_tar.tar` | 90 MB | C++ helper, installed as a launch agent |
| `Frameworks/libLiteRt.dylib` | 12 MB | LiteRT runtime |
| `Frameworks/libLiteRtMetalAccelerator.dylib` | 17 MB | LiteRT Metal GPU accelerator |
| `Frameworks/libLiteRtWebGpuAccelerator.dylib` | 32 MB | LiteRT WebGPU (Dawn) accelerator |
| `Frameworks/libLiteRtTop*Sampler.dylib` | 34 MB | Top-K samplers for Metal and WebGPU |
| `Frameworks/FlutterMacOS.framework` | 15 MB | Flutter engine |
| `Resources/GoogleUpdater.zip` | 13 MB | Google auto-updater (Keystone) |

`Info.plist` asks for microphone access (`NSMicrophoneUsageDescription`) and
for system audio capture (`NSAudioCaptureUsageDescription`). It also contains
a Google Sign-In client ID.

### 3.3 Architecture

```
Flutter UI (Dart)
  |-- LiteRT-LM through Dart FFI (package litert_lm): Gemma 4 generation
  |-- ASR: on_device_speech_transcription_service, VAD through FFI
  |-- SQLite through drift: documents, transcripts, notes, graph RAG
  |-- Google APIs: Drive, Docs, Slides, Calendar
  |
  +-- gRPC over 127.0.0.1 --> ForesightHelper (C++ launch agent)
                                 RagService: embeddings, chunking, index, retrieval
```

gRPC methods in the helper
(`research.xeno.labs.foresight.launch_agent.proto.RagService`):
`Ping`, `UpdateModelPaths`, `ChunkText`, `TruncateText`, `Encode`,
`ComputeEmbedding`, `ComputeImageEmbedding`, `ComputeAudioEmbedding`,
`IndexDocument`, `DeleteDocument`, `RetrieveContext`.

gRPC methods for the cloud backend (`ai_edge.foresight.service.ForesightService`):
`CheckServiceStatus`, `FetchExternalWebsite`, `GetSharedConfig`,
`ResolveGoLinks`, `UnsubscribeFromFiles`, `UpdateFcmToken`. A second service
(`CalendarEventsSubscriberService`) subscribes to calendar webhooks at
`https://odml.pa.googleapis.com/v1/calendar/notification`.

### 3.4 App modules (from Dart package paths)

The app has about 250 Dart source files. The main groups are:

| Group | Example files | Purpose |
|---|---|---|
| `asr/` | `multiplexed_speech_stream.dart`, `vad_detector_ffi.dart`, `audio_format_converter.dart` | Mixes the audio sources, VAD, transcription |
| `model/query_detector/` | `query_detector_on_device.dart`, `speaker_id_manager.dart`, `transcript_buffer.dart`, `text_deduplicator.dart` | Live Assistance: finds questions in the transcript |
| `rag/` | `rag_orchestrator.dart`, `rag_reranker.dart`, `remote_tokenizer.dart` | Retrieval and reranking |
| `database/` | `graph_rag_service.dart`, `document_graph_service.dart`, `event_transcript.dart`, `note.dart` | SQLite schema, graph RAG |
| `service/` | `notes_enhancer.dart`, `insight_service.dart`, `map_reduce_executor.dart`, `transcript_summary_manager.dart`, `local_file_discovery_service.dart`, `hotkey_service.dart` | Notes, insights, long-transcript summaries, file discovery |
| `file_parsers/` | PDF, DOCX, PPTX, XLSX, ODT, EPUB, RTF, HTML, MD, TXT, XML, ZIP | Text extraction |
| `nlp/` | `add_calendar_event_intent_*`, `compose_gmail_intent_*`, `send_sms_intent_*` | Action intents from the transcript |
| `model_downloader/` | `chunk_downloader.dart`, `download_tracker.dart` | Model download |
| `encryption.*` | `macos_encryption_service.dart` | Model decryption |
| `plugins/` | `audio_capture`, `hotkey`, `work_manager`, `firebase_message` | Native macOS plugins |

### 3.5 Models

- **LLM.** Gemma 4 through LiteRT-LM. The string `insightInferenceModel` and a
  `model_selection_page.dart` show that the user can select a model.
- **Embeddings.** EmbeddingGemma 2 (`litert.lm.proto.EmbeddingGemmaV2`), with
  text, image, and audio embedding calls.
- **Speech recognition.** An ODML speech pipeline
  (`third_party/odml/audio/transcription/...`) with a VAD segmenter
  (`webrtc_rnn_vad`) and a `LiteRtLmSpeechProcessor`. The LiteRT-LM model list
  contains small speech models such as `GEMMA4_200M_USM_300M`,
  `GEMMA4_200M_USM_75M`, and `GEMMA4_42M_USM_24M` (a small Gemma 4 decoder with a
  USM speech encoder). The binary contains the full LiteRT-LM model list, so
  we do not know which entry Foresight uses. USM models are not public.
- **Model delivery.** `AssetManager` downloads the models from
  `odml.pa.googleapis.com` (OAuth scope `https://www.googleapis.com/auth/aiedge`).
  The app then decrypts them with Tink (`HybridDecrypt`) into temporary files.
  The models are not usable outside the app, and we must not try to use them.

### 3.6 Network use

The inference is local. The app still uses these network services:
Google Sign-In, Drive, Docs, Slides, Calendar, Firebase Analytics and Cloud
Messaging, App Check, the model download API, and Google Update.

## 4. Linux availability of the building blocks

Checked on 2026-10-12 with the Hugging Face API, PyPI, and the GitHub API.

| Component | Status | Evidence |
|---|---|---|
| Gemma 4 E2B / E4B | Public, Apache 2.0, not gated | HF `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it` |
| Gemma 4 in Ollama | Available | <https://ollama.com/library/gemma4> (HTTP 200) |
| EmbeddingGemma 2 | Public, Apache 2.0, not gated | HF `google/embeddinggemma-2` (`feature-extraction`) |
| EmbeddingGemma 2 GGUF | Available | HF `unsloth/embeddinggemma-2-GGUF` |
| LiteRT-LM | Linux x86_64 wheel | PyPI `litert-lm-api` 0.18.0, `manylinux_2_27_x86_64` (47 MB) |
| LiteRT | Linux x86_64 wheel | PyPI `ai-edge-litert` 2.3.0 |
| Flutter on Linux | Supported | Flutter desktop |
| USM speech models | Not public | No public weights found |

### 4.1 Model facts (from the research agent, not checked again)

- **EmbeddingGemma 2:** 740M in total (130M text backbone, 140M embedder,
  170M vision encoder, 300M audio encoder). Output is 768 dimensions, with
  Matryoshka truncation to 512, 256, or 128. Context is 8,192 tokens. LiteRT-LM
  lists 740M, 440M (text and image), and 270M (text) variants.
  Sources: <https://huggingface.co/google/embeddinggemma-2>,
  <https://developers.google.com/edge/litert-lm/overview>.
- **Gemma 4:** E2B (2.3B effective, 2.9 GB at Q4_0) and E4B (4.5B effective,
  4.5 GB at Q4_0) accept audio input, with a 30-second limit for each clip.
  Larger variants: 12B, 26B A4B (MoE), and 31B. The RAM figures do not include
  the KV cache. Source: <https://huggingface.co/google/gemma-4-E4B-it>.
- **llama.cpp:** Gemma 4 audio support came with PR #21421. The llama.cpp docs
  call audio support "highly experimental". Source:
  <https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md>.
- **MediaPipe LLM Inference** is in maintenance mode, and it does not support
  Linux. Google tells users to move to LiteRT-LM. Source:
  <https://developers.google.com/edge/mediapipe/solutions/genai/llm_inference>.

### 4.2 Speech recognition and audio capture on Linux

- **Speech recognition:** whisper.cpp, faster-whisper, or NVIDIA
  Parakeet/Canary. Gemma 4 audio input works for 30-second chunks.
- **Diarization:** pyannote Community-1 (pyannote.audio 4.x). It needs PyTorch.
- **System audio:** record two streams, the microphone (`@DEFAULT_SOURCE@`) and
  the monitor of the default sink (`@DEFAULT_MONITOR@`). This works on
  PulseAudio and on PipeWire through `pipewire-pulse`. With two streams, the
  app can label speech as "Me" or "Others" without diarization.

## 5. Similar open-source projects

| Project | Linux | Notes |
|---|---|---|
| Meetily (<https://github.com/Zackriya-Solutions/meeting-minutes>) | Build from source | Mic and system audio, Whisper or Parakeet, Ollama. No RAG. MIT. |
| Anarlog, formerly Hyprnote (<https://github.com/fastrepl/anarlog>) | Download available | Local speech recognition, Ollama. Linux quality not checked. MIT. |
| Scriberr (<https://github.com/rishikanthc/Scriberr>) | Docker | Works on files, not live. Diarization, chat. Development paused. MIT. |

No project combines a live meeting notetaker with a multimodal local knowledge
base. That combination is the gap that a Linux version can fill.

## 6. Items not verified

- Which speech model and which Gemma 4 variant Foresight uses.
- LiteRT-LM performance and GPU support on Linux x86_64.
- GGUF and llama.cpp support for the audio and vision encoders of
  EmbeddingGemma 2.
- Audio input for Gemma 4 in Ollama.
- Multi-window and tray support in Flutter on Linux.
- Press sources: we did not read each article. Sources found by the research
  agent:
  <https://www.notebookcheck.net/Google-AI-Edge-Foresight-Offline-AI-for-meeting-notes-on-Mac.1418337.0.html>,
  <https://www.howtogeek.com/googles-new-ai-note-taker-doesnt-need-the-cloud/>,
  <https://windowsreport.com/google-foresight-is-a-new-local-ai-note-taking-app-that-works-offline/>.

## 7. Decisions so far

| Topic | Decision |
|---|---|
| Target hardware | Any Linux x86_64 PC with 16 GB RAM, 8 cores, and AVX2. A GPU is optional. |
| UI | Qt 6 with PySide6 and QML (native app, tray, floating panel, Wayland portals) |
| Inference | Embedded llama.cpp (`llama-cpp-python`) for Gemma 4. faster-whisper for speech recognition. A spike decides how to run EmbeddingGemma 2. Optional Ollama backend. |
| v1 scope | (A) meeting capture and live transcript, (B) notes enhancement, (C) local knowledge base with RAG chat. Live assistance comes later. |
| Knowledge sources | Local files only. No accounts. |
| Language | English only |
| Speakers | "Me" (mic) and "Others" (system audio). No diarization in v1. |
| Storage | SQLite with `sqlite-vec` and FTS5 |
| Process model | A headless engine process and a Qt UI client, connected by JSON-RPC over a Unix socket |
| Working name | "Lookout". We do not use the name "Foresight". |
