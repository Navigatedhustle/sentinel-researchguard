# Sentinel / ResearchGuard

A research workspace for synthetic policy curricula, strict evidence-conditioned decisions, adapter training and challenge evaluations. Beat the Guard consumes its V6 adapter.

**Maturity:** Research archive with multiple curriculum and evaluation generations. Source compiles, but training/evaluation runs were not restarted and no accuracy or current adapter reproducibility claim is made. The old training container was stopped before this task.

## How it works

Synthetic curriculum builders → label/evidence causality audits and holdout challenges → model training scripts → adapter checkpoints → strict-JSON evaluations. `rag/` provides policy-retrieval experiments. Historical source experiments are retained under their original version directories.

## Getting started

Read [installation, configuration and shutdown](docs/OPERATIONS.md) and [model/data retrieval](docs/MODELS.md). Review [dependency evidence](docs/DEPENDENCIES.md) before choosing an environment. The safe review launcher uses loopback and does not install services or load models.

## Security and data

Read [security and safe updates](SECURITY.md) and [ownership/license status](docs/LICENSE-STATUS.md). Runtime originals are preserved separately. This is a sanitized source snapshot with new history: no account state, business records, model weights, unknown media assets or original logs/history are distributed. Third-party notices included in this copy remain applicable.

## Verification

See [verification evidence and limitations](docs/VERIFICATION.md). Tests distinguish synthetic/mocked workflows from live integrations. Original demo screenshots and document recordings are excluded; use the synthetic workflow in the operations guide for your own reviewed demonstration.

## Roadmap

Select a canonical V6 pipeline, pin environment and dataset-generation seeds, separate training/evaluation contamination checks and publish synthetic-only reproducibility evidence after a dedicated rerun.

For route/source details see [architecture](docs/ARCHITECTURE.md).
