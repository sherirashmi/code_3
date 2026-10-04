# Forward operators and their invertible versions

This note lists what each invertible model takes from the forward operator it
is built from (`erp_forward_operators/`), and what had to change. There are
two invertible families.

| Family | Folder | Idea | Inverse |
|---|---|---|---|
| **Invertible DeepONet (iDON)** (Kaltenbach et al. 2023) | `erp_invertible_deeponet/` | RealNVP branch (design → Q coefficients) × trunk (Q fixed basis functions of f). ERP = Ψ b + ψ₀ | **exact**: projection b* = Ψᵀ(y − ψ₀)/F, then RealNVP⁻¹; samples from a Gaussian around b* |
| **Coupling-flow operators** (Long et al. 2024, iFNO) | `erp_invertible_operators/` | lift P (design → latent along f) → invertible coupling stack whose gate L is the architecture's own layer → readout Q (ERP) | only the coupling stack is exact; the inverse lift P′ (ERP → latent) and readout Q′ (latent → design) are learned; a β-VAE gives several designs |

## 1. Differences that apply to every invertible model

| | Forward operators (final models) | Invertible models |
|---|---|---|
| Task | design → ERP | design → ERP **and** ERP → several designs, in one network |
| Dataset | `100k`: 3 resonators, 150 plate modes, m, k, f_t, x, y all varied | `200k_2res_18modes`: 2 resonators, 18 plate modes, m, f_t, x, y varied (k = m(2πf_t)²) |
| Design representation | z-scored [m, k, f_t, x, y] per resonator, an unordered set | 8 numbers [m, f_t, x, y] per resonator, logit-bounded to the physical ranges and sorted by f_t; k derived, so every predicted design is physical and consistent |
| Resonator encoder | set encoder (shared MLP, mean + max pooling) **+** f_t-sorted encoder, fused | f_t-sorted encoder only (coupling family); iDON: the 8 sorted numbers go straight into the RealNVP |
| Position features | plate mode shapes sin(mπx/L_x), sin(nπy/L_y) + products; detuning (f − f_t)/σ_f | the same (coupling family); iDON: none — its trunk sees only f |
| Where the design may enter | anywhere (FiLM, poles, attention, … can depend on the design at every layer) | coupling family: **only in the lift P** — the gate L also runs in the inverse direction, where the design is unknown; iDON: **only in the branch** — the trunk must be design-independent, otherwise the inverse is no longer a projection |
| Loss | ERP MSE + slope + peak terms | coupling family: 3 stages (forward MSE + inverse MSE + lift/readout round trips + cycle/alignment; β-VAE; joint); iDON: forward MSE + Huber(inverse design error) + 0.1 Huber(padding latent) |
| Training | 200 epochs | coupling: 20 + 25 + 15 epochs; iDON: 150 epochs, stopped at epoch 50 if the validation loss stops improving |
| Size | ≈ 110–160k parameters | coupling: ≈ 60–190k; iDON: ≈ 1.3–1.9M (mostly the RealNVP) |

## 2. Per architecture

"Gate L" = the layer inside every coupling block (it acts on half of the
latent, along the frequency axis, and must not see the design). "iDON trunk" =
how the basis functions of the invertible DeepONet are built with that
architecture's layers (a function of f only).

| Forward model | Forward core | Coupling version: lift P (design enters here) | Coupling version: gate L | iDON trunk |
|---|---|---|---|---|
| **NN** | plain MLP on [flattened design, f] | — (no architecture-specific layer to reuse) | — | — |
| **DON** | encoder → branch coefficients × trunk basis (FiLM from the design), 4 terms, refinement | — (the invertible DeepONet *is* the invertible DON) | — | **Q64 / Q128**: Fourier features + MLP (the DeepONet trunk); no FiLM (it would make the basis design-dependent) |
| **DNO** | encoder + frequency encoder + resonance query → FiLM residual blocks (γ, β from design + detuning) → refinement | **iDNO**: [c, emb(f), q(f)] → Linear (DNO's lift) | FiLM residual block + refinement, **FiLM driven by a frequency embedding** instead of the design | **Q64-DNO**: FiLM residual blocks with FiLM from a frequency embedding, + refinement |
| **FNO** | encoder + resonance query → Fourier blocks (spectral conv + local conv) | **iFNO**: [c, q(f), f] → Linear | Fourier layer (spectral conv on the lowest modes + local conv), padded FFT | **Q64-FNO**: Fourier blocks along f |
| **DCO** | branch (encoder) + trunk MLP(f) + resonance query concatenated → residual MLP blocks → refinement | **iDCO**: [branch, trunk(f), q(f)] → Linear (DCO's lift) | residual MLP block + refinement | **Q64-DCO**: residual MLP blocks + refinement |
| **GNO** | graph of resonators (message passing) → per-f kernel + attention over resonators → refinement | **iGNO**: node lift + message passing + kernel/attention per f → Linear | pointwise MLP + refinement (message passing acts on resonators, not on f) | — (its core acts on the resonators, which a trunk never sees) |
| **STO** | resonator tokens (self-attention) → cross-attention from f to resonators with detuning bias → frequency mixer | **iSTO**: tokens + detuning-biased cross-attention → Linear | frequency mixer (local + dilated Conv1d) | — (same reason as GNO) |
| **SIREN** | encoder; [f, q(f)] → sine layers modulated by the design → refinement | **iSIREN**: [f, q(f)] → modulated sine layers (FiLM from c) → Linear | unmodulated sine layer + refinement | **Q64-SIREN**: sine layers on f (replacing the Fourier features) + refinement |
| **WNO** | encoder + resonance query → multi-level Haar wavelet blocks | **iWNO**: [c, q(f), f] → Linear | multi-level Haar wavelet block, unchanged | **Q64-WNO**: Haar wavelet blocks along f |
| **LNO** | encoder → design-dependent poles/residues → pole features r/(s − p) + conj at s = i f + query → refinement | **iLNO**: design-dependent poles → pole features + q(f) + f → Linear | **learned, design-independent poles** → pole features + pointwise transform, + refinement | **Q64-LNO**: learned, design-independent poles (resonance-shaped basis) + Fourier features + refinement |

Notes:

* GNO and STO have no set/sorted encoder in either direction: their resonators
  are graph nodes / attention tokens, which are permutation invariant by
  construction.
* The common thread of every change: a layer that is **conditioned on the
  design** in the forward model (FiLM in DON/DNO/SIREN, poles in LNO,
  cross-attention in STO/GNO) is kept in the part that sees the design (lift P
  or branch) and replaced by a **design-independent** version in the part that
  must also work backwards (gate L or trunk).
* iDON trunks are trained on the same data as Q64, so only the trunk differs
  (Q64-FNO and Q64-DCO were no better than the plain MLP trunk; the basis size Q
  mattered more).

## 3. Commands

    # invertible DeepONet, one or more variants
    python -m erp_invertible_deeponet.train Q64-DNO Q64-WNO Q64-LNO Q64-SIREN --epochs 150
    python -m erp_invertible_deeponet.evaluate Q8 Q64 Q128 Q64-FNO Q64-DCO Q64-DNO Q64-WNO Q64-LNO Q64-SIREN

    # coupling-flow operators (keys: 1 iFNO, 2 iDCO, 3 iGNO, 4 iDNO, 5 iWNO, 6 iLNO, 7 iSIREN, 8 iSTO),
    # sorted encoder + plate mode shapes by default
    python -m erp_invertible_operators.run_dataset 200k_2res_18modes 1 2 3 4 5 6 7 8
