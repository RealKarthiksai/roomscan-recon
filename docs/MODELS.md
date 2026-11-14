# Models

| Model | HF id | Licence | Use |
|---|---|---|---|
| Depth-Anything-V2 Metric-Indoor **Small** (default) | `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` | **Apache-2.0** | Phase 3 mono depth (metric, Hypersim-trained) |
| Depth-Anything-V2 **Small** (relative) | `depth-anything/Depth-Anything-V2-Small-hf` | Apache-2.0 | Step A NYU reproduction; relative-depth variant |
| Depth-Anything-V2 Metric-Indoor Base / Large | `…-Metric-Indoor-Base-hf` / `…-Large-hf` | **CC-BY-NC-4.0** | opt-in only (`--model base|large`); not used for product defaults |

Licence source: Depth-Anything-V2 README ("Small model is under the Apache-2.0 license.
Base/Large/Giant models are under the CC-BY-NC-4.0 license."). The HF model cards carry no
licence field. The exact weights revision used is recorded per run in
`depth_alignment.json → model.revision`.
