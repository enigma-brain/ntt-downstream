# Third-party licences

The code in this repository is MIT-licensed ([`../LICENSE`](../LICENSE)). The files
below adapt third-party code and keep its licence; each file's header names its
upstream. This folder holds the upstream licence texts and copyright notices.

| Licence file | Upstream | Adapted in |
|---|---|---|
| [`torch_brain-Apache-2.0.txt`](torch_brain-Apache-2.0.txt) | [torch_brain](https://github.com/neuro-galaxy/torch_brain), Copyright 2025 NeuroGalaxy | `src/ntt_downstream/decoding/backends/poyo/{poyo,model,train_wrapper,multi_session_setup,flash_attn_varlen}.py` |
| [`hiera-Apache-2.0.txt`](hiera-Apache-2.0.txt) | [Hiera](https://github.com/facebookresearch/hiera), Copyright (c) Meta Platforms, Inc. and affiliates | `src/ntt_downstream/encoding/models/enc/hiera/{modded_hiera,utils}.py` |
| [`mar-MIT.txt`](mar-MIT.txt) | [MAR](https://github.com/LTH14/mar), Copyright (c) 2024 Tianhong Li | `src/ntt_downstream/encoding/models/enc/losses/diffloss.py`, `losses/diffusion/*`, `hiera/mar_model.py` |
| [`openai-diffusion-MIT.txt`](openai-diffusion-MIT.txt) | [guided-diffusion](https://github.com/openai/guided-diffusion), [improved-diffusion](https://github.com/openai/improved-diffusion), [glide-text2im](https://github.com/openai/glide-text2im), Copyright (c) 2021 OpenAI | `src/ntt_downstream/encoding/models/enc/losses/diffusion/*`, `losses/diffloss.py` |
| [`ear-MIT.txt`](ear-MIT.txt) | [EAR](https://github.com/shaochenze/EAR), Copyright (c) 2024 Tianhong Li (as stated in EAR's licence file) | `src/ntt_downstream/encoding/models/enc/losses/energy_loss.py` |
| [`open-retina-MIT.txt`](open-retina-MIT.txt) | [open-retina](https://github.com/open-retina/open-retina), Copyright (c) 2025 open-retina | `src/ntt_downstream/encoding/models/enc/conv/*` |
| [`ldns-MIT.txt`](ldns-MIT.txt) | [LDNS](https://github.com/mackelab/LDNS), Copyright (c) 2024 mackelab | `src/ntt_downstream/encoding/models/enc/losses/ldns_denoiser.py` |

`TimestepEmbedder`, `FinalLayer`, `modulate` and `create_diffusion` originate in
[DiT](https://github.com/facebookresearch/DiT) (CC BY-NC 4.0); this repository uses
them as redistributed by MAR under the MIT licence.
