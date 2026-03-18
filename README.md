# ICNO （CMAME 2026）
Information-Coupled Neural Operator for Computational Mechanics and Parametric PDEs[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0045782526001258)


The main contributions of ICNO are threefold:

- We elaborate an efficient Cross-frequency Fourier Attention mechanism that couples multi-scale information in the Wavelet domain. The proposal simultaneously reduces spatial and temporal complexities, delivering superior efficacy and efficiency compared to current methods.
- We present a wavelet block to separate and reconstruct information components, along with an interleaved module to balance global-local information. By coupling spectral learning with attention computation, the method is able to capture a richer set of spectral-spatial features.
- We systematically explore cross-spectral coupling mechanisms and distill an effective integration recipe that consistently enhances operator learning.

<p align="center">
  <img src="pic/ICNO.webp"
       style="width:100%;  object-fit:cover;">
<br><br>
<b>Figure 1.</b> Overview of ICNO.
</p>

## Results
<p align="center">
  <img src="pic/result.webp"
       style="width:100%;  object-fit:cover;">
<br><br>
<b>Figure 2.</b> Results on six standard benchmarks.
</p>
