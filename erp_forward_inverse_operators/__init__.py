"""Invertible forward+inverse operators for the ERP config<->spectrum
problem: one shared coupling-flow scaffold (coupling.py, common.py),
adapted from Long, Xu, Yuan, Yang & Zhe, "Invertible Fourier Neural
Operators for Tackling Both Forward and Inverse Problems"
(arXiv:2402.11722), instantiated three times with a different forward
operator's own characteristic layer as the coupling blocks' gate function:

  - ifno.py: gated by an FNO Fourier layer (the paper's own choice).
  - idco.py: gated by DCO's residual-MLP-plus-frequency-refinement pair.
  - igno.py: gated by a GNO-derived pointwise-MLP-plus-frequency-refinement
    pair, with GNO's own graph message-passing and attention query
    conditioning the lift (see igno.py's docstring for why graph
    message-passing itself can't be the gate function).

See registry.py for the "any of the three" driver used by train.py.
"""
