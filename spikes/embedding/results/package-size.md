# Installed size of each backend (Python packages only, no models)

| Backend | Size |
|---|---|
| sentence-transformers + torch CPU | 1.1 GB (torch 2.14.1+cpu) |
| llama-cpp-python | 120 MB |
| litert-lm-api | 126 MB |

Note: the plan command installs `torch 2.14.1+cu130` with the NVIDIA CUDA libraries, because pip took torch from PyPI. That install is 5.5 GB. The 1.1 GB figure comes from a second install with `torch==2.14.1+cpu` and `--index-strategy unsafe-best-match`.
