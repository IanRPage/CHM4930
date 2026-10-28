# MC-Dropout

Every pIC50 prediction gets an interval $[\hat y - q\sigma,\ \hat y + q\sigma]$ per target (`pIC50_BACE1`, `pIC50_EGFR`) at 90% and 95% coverage. MC-Dropout [^16] supplies $\sigma$ and conformal calibration [^19] supplies $q$.

## How it's Computed

For each compound:

1. **MolFormer once.** The frozen SMILES encoder runs a single time and its embedding is reused below.
2. **Point prediction** $\hat y$: one normal eval-mode pass.
3. **Spread** $\sigma$: the fingerprint encoder, GIN encoder, projection, fusion and regression head are re-run $T = 50$ times with dropout on. $\sigma$ is the standard deviation of the $T$ predictions. But the normalization layers stay in eval mode and modality dropout is off.
4. **Interval** $\hat y \pm q \cdot \max(\sigma, \sigma_{\text{floor}})$.

## Conformal calibration

On the calibration split, for each target and coverage $1-\alpha$ (σ-normalized scores and the $(n+1)$ quantile follow [^19]):

$$
s_i = \frac{|y_i - \hat y_i|}{\max(\sigma_i,\ \sigma_{\text{floor}})}, \qquad q = s_{(k)},\quad k = \lceil (n+1)(1-\alpha) \rceil
$$

where $s_{(k)}$ is the $k$-th smallest of the $n$ scores. This is the $\lceil (n+1)(1-\alpha) \rceil / n$ quantile, not plain $1-\alpha$. This is what makes the coverage guarantee hold. If $k > n$ there are too few labeled compounds and $q = \infty$.

$\sigma_{\text{floor}}$ is the 5th percentile of the target's calibration $\sigma$, so a few near-zero $\sigma$ values can't blow up $q$. It's also applied at inference.

## Reading an interval

- **Narrow:** indicates dropout sub-networks agree on this compound, so the model is confident.
- **Wide:** indicates dropout sub-networks disagree, so treat the prediction as less reliable. Could happen if the compound is unlike the training set.
- Width is $2q\sigma$, and $q$ is shared by all compounds. So $\sigma$ is what ranks compounds by confidence.

## What the coverage means

- A 90% interval means about 90% of compounds' true pIC50s fall inside their intervals **on average, over compounds**. NOT a 90% guarantee for any single compound.
- It assumes calibration and test compounds are exchangeable. Our benchmark uses a scaffold split, meaning coverage is weaker for compounds unlike the training set.
- Coverage may be more optimistic in our case because val split is used for early stopping _and_ classification calibration.
- $\sigma$ only means something on trained weights.

## Ensemble intervals

The 5-seed ensemble (`mufu/ensemble.py`) predicts $\bar y = \frac1S\sum_s \hat y_s$ and combines each seed's MC-Dropout $\sigma_s$ with the spread between seeds:

$$
\sigma_{\text{comb}}^2 = \frac1S\sum_s \sigma_s^2 + \frac{1}{S-1}\sum_s (\hat y_s - \bar y)^2
$$

$q$ and $\sigma_{\text{floor}}$ are fitted on val exactly as above, using $\bar y$ and $\sigma_{\text{comb}}$. They are saved to `checkpoints/<model_name>_ensemble.json` with each member's sha256, and loading fails if a member was added, removed or retrained. $q$ is also only valid for the same $T$ and dropout rates.

Test-split coverage and widths from the SCRUM-90 seeds are in EXP-009 (H6).

[^16]: Gal, Y., & Ghahramani, Z. (2016, October 4). _Dropout as a bayesian approximation: Representing model uncertainty in Deep learning_. arXiv.org. https://doi.org/10.48550/arXiv.1506.02142
[^19]: Angelopoulos, A. N., & Bates, S. (2021). _A gentle introduction to conformal prediction and distribution-free uncertainty quantification_. arXiv.org. https://doi.org/10.48550/arXiv.2107.07511
