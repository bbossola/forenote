# GGUF image embedding with the llama.cpp CLI

- llama.cpp release: b11565 (the zip asset does not exist; the asset is `llama-b11565-bin-ubuntu-x64.tar.gz`)
- `llama-embedding` has image options: no (the release has no `llama-embedding` binary; `llama-server` and `llama-mtmd-cli` have `--mmproj`)
- Image embedding command exit code: not run with `llama-embedding`. `llama-server --embeddings --mmproj` returned HTTP 200 for all 4 images.
- Output: 768 numbers: yes (all 4 images, L2 norm 1.0)
- Text embedding loads in `llama-embedding`: not testable (no binary). It loads in `llama-server --embeddings`: yes, 768 numbers.
- Notes: The tag `latest` on GitHub is `v0.6.0` and has no binaries, so the release `b11565` was used. The server needs the media marker from `/props` in `prompt_string`, with the image in `multimodal_data` as base64 on `POST /embeddings`. The image vision token budget is capped to 256 by n_ubatch 512. Quality against the reference was not measured.
