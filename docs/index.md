# Documentation

## Start

- install and basic commands: [`../README.md`](../README.md)
- training presets and strict parity runs: [`replication.md`](replication.md)

## Configuration and Model Behavior

- backbone families, source/weight resolution, embedding sharing: [`model.md`](model.md)
- normalization options for rope (`post`, `keel`): [`norm-strategy.md`](norm-strategy.md)

## Data and Objective

- dataset loading, packing, doc-blocking masks, collator behavior: [`data.md`](data.md)
- RTD flow, loss terms, decoupled phases: [`objective.md`](objective.md)

## Runtime and Export

- accelerate/FSDP2, compile, resume, checkpoint/export behavior: [`fsdp2.md`](fsdp2.md)

## Engineering Notes

- runtime compatibility notes and known limitations: [`dev/index.md`](dev/index.md)
- deferred tasks: [`roadmap.md`](roadmap.md)
