"""Invertible forward+inverse operators for the ERP config<->spectrum
problem: one shared coupling-flow scaffold (coupling.py, common.py),
adapted from Long, Xu, Yuan, Yang & Zhe, "Invertible Fourier Neural
Operators for Tackling Both Forward and Inverse Problems"
(arXiv:2402.11722), instantiated with a different forward operator's own
characteristic layer as the coupling blocks' gate function:

  - ifno.py: gated by an FNO Fourier layer (the paper's own choice).
  - idco.py: gated by DCO's residual-MLP-plus-frequency-refinement pair.
  - igno.py: gated by a GNO-derived pointwise-MLP-plus-frequency-refinement
    pair, with GNO's own graph message-passing and attention query
    conditioning the lift (see igno.py's docstring for why graph
    message-passing itself can't be the gate function).
  - idno.py: gated by DNO's FiLM residual block (FiLM from a frequency
    embedding) + refinement.
  - iwno.py: gated by WNO's multi-level Haar wavelet block.
  - ilno.py: gated by learned, design-independent pole features + refinement;
    LNO's design-dependent poles in the lift.
  - isiren.py: gated by a sine layer + refinement; SIREN's modulated sine
    stack in the lift.
  - isto.py: gated by STO's frequency mixer; resonator tokens and
    detuning-biased cross-attention in the lift.

See registry.py for the driver used by train.py, and FORWARD_VS_INVERTIBLE.md
for what each model keeps from, and changes with respect to, its forward
operator.
"""
