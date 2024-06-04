Chi, Cheng, et al. "Diffusion policy: Visuomotor policy learning via action diffusion." arXiv preprint arXiv:2303.04137 (2023).

## Point
diffusion model is expressive enough to output action sequence instead of single action. This is useful for temporal consistency in action sequence. Also, the denoising process makes the model robust to multi-modality.

## prerequisites
- learning a latent space representation of $x$ so that the denoising process can reproduce the original data
- Essentially, learning a compressed representation. Thus, self-supervised learning is adopted.

## How to apply the diffusion model to policy learning
- output $x$ is an action sequence
    - focus on the temporal consistency of the action sequence
    - thus $x$ is a sequence of actions instead of a single action
    - input is $T_o$ frames of observations. Output is $T_p$ frames of action prediction.
    - $T_a$ is the action execution frames (What is this??!)
- denoising conditioned on image observation

## Design choice
- Both CNN and transformer are used for the encoder and decoder
    - recommendation: try first CNN
    - CNN-policy refers to  Tancik, Matthew, et al. "Fourier features let networks learn high frequency functions in low dimensional domains." Advances in neural information processing systems 33 (2020): 7537-7547.
    - Due to inductive bias, CNN-based policy does not work well when action is abruptlly changed (e.g., requiring velocity control). Probably due to inductive bias in temporal convolution. 

## Why good
- denoising stochasticity makes it robust to multi-modality
- chunk (action sequence) prediction
    - temporal consistency (no jitter)
    - robustness to idle action

## Speed
- 10hz using NVIDIA 3080
