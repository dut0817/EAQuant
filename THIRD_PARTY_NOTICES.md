# Third-party notices

This repository incorporates and modifies source code from the following
projects. EAQuant-specific changes include the shared evidence objectives,
MedMix/evidence-cache integration, and paper evaluation entry points.
Only backend components required by the EAQuant paper workflow are retained.

## OmniQuant

- Upstream: <https://github.com/OpenGVLab/OmniQuant>
- Imported revision: `feffe8e`
- License: MIT; retained at `backends/omniquant/LICENSE`

## OSTQuant

- Upstream: <https://github.com/BrotherHappy/OSTQuant>
- Imported revision: `ab64362`
- License: Apache License 2.0; retained at `backends/ostquant/LICENSE`

The notices above apply to the backend-derived files. The root MIT license
applies to original EAQuant code except where a retained third-party license
states otherwise.

The imported revisions identify the upstream bases. They do not identify the
later uncommitted research modifications or constitute immutable snapshots of
the paper artifacts; that reproducibility scope is described in the root
`README.md`.
