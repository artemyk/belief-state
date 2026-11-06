# LessWrong figures

Publication figures for **Generalized belief states in transformers**.

This branch contains figure assets and supporting checkpoints and has independent history from the code branches.

The PNGs and animated GIF are copied from the padded publication exports. Transparent horizontal margins preserve the draft's intended relative figure widths when the images fill the post column. The reference width is 720 pixels; source pixels are not rescaled.

The GIF retains all 900 frames, frame timing, looping, and transparent margins.

For stable Markdown embeds, use raw.githubusercontent.com URLs pinned to the commit containing these files.

## RoPE and visible-BOS transformer figures

The three transformer figures use ordinary softmax attention with visible BOS. The decoder reads normalized attention outputs and current-token coordinates; the affine encoder writes every KV entry and the current token. [Metrics, fitting parameters, and checksums](./figures/transformer-fit.json) record the evaluated fit.

- [Transformer checkpoint](./checkpoints/rope-visible-bos/teacher.pt): config and model state dictionary.
- [Decoder/encoder checkpoint](./checkpoints/rope-visible-bos/abstraction.pt): load with `GBSAbstraction.load` and `BeliefStateAdapter.from_checkpoint` from [artemyk/gbs](https://github.com/artemyk/gbs).
