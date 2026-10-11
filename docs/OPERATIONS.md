# Installation and operation

## Environment

Recorded development hardware: NVIDIA GB10 ARM64, nominal 128 GB unified memory (about 121 GiB usable), Ubuntu 24.04.4 LTS, NVIDIA driver 580.173.02 and CUDA driver support 13.0. These are recorded facts, not universal minimum requirements. Tests used existing project environments on that machine; fresh dependency installation on another machine was not performed. Model-heavy modes need enough free unified memory and storage for the assets. Start heavyweight applications sequentially and check available resources first.

Python target: 3.12. Dependencies: source manifests and `docs/DEPENDENCIES.md`. Exact installed versions are recorded where available; an installed-environment freeze is not proof every optional package is required. Use project-scoped environments and do not replace the host NVIDIA driver/toolkit.

```sh
python3.12 -m venv .venv
# Select the recorded NVIDIA container/runtime from docs/MODELS.md for training; do not start training as setup.
```

## Configuration, startup and shutdown

This repository is an inspectable research-source archive, not a turnkey inference service. Dependencies vary by version and are listed in `docs/DEPENDENCIES.md`. Do not run training builders or evaluations until their input/output paths and experiment budgets are reviewed. No training work was authorized as part of portfolio preparation.

This is a source/research/integration utility. No general automatic service startup is supplied. Review the repository's entry scripts and upstream requirements before executing them; see README for the safe checks performed.

## Troubleshooting

If imports fail, activate the correct project environment and inspect dependency manifests. If a port is occupied, stop your own review process or choose another review port; do not stop an unrelated runtime. If model readiness fails, verify local assets, model descriptors, license acceptance and loopback endpoints. Missing excluded assets are an honest setup error; no cloud inference fallback is configured by the publication launcher. Empty directories and generated state are recreated during setup/startup.

For frontend missing-directory errors, build frontend assets or create the required empty output directory; do not commit generated bundles. For optional model failures, distinguish control-plane/unit-test success from verified live inference. Logs, exports, databases, screenshots and cache indexes belong in private runtime storage and remain ignored. Restore the original project's service state only if you intentionally changed it; portfolio preparation never did.
