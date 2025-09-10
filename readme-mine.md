
`/configs/infer.sh` has the instructions for inference


`runs/infer_minilm_v1/inference_report_v1.md` this is where the output of the inference is. 


## Guest Feature Engineering
`train_regressor_minilm_guestfeat_v1`
`train_regressor_minilm_guestnorm_v1`

- For guestnorm, with each unique guest, the log normalized views mean is calculated and that is the value. If the guest is unseen at time of inference it gets global mean. 

## Results of Guest Feature & Best Model Right Now
### Training Report — Guest Feature (v1)

CSV: /Users/matthewatkinson/Apps/shorts-model_v3/data/training-data_v4.xlsx
Model: sentence-transformers/all-MiniLM-L6-v2 | Dim: 384 | Device: mps

#### Cross-validated metrics (on y_log)

- R^2 mean ± std: 0.308 ± 0.217
- Spearman mean ± std: 0.660 ± 0.116



## Here’s the practical difference between the two guest-aware training runs you have:

What they share
•  Embeddings: MiniLM (sentence-transformers/all-MiniLM-L6-v2)
•  Label base: y_log = log1p(view_count)
•  CV: 5-fold with leakage control for guest stats (guest means computed on train split only)
•  Artifacts include the trained Ridge regressor and fold-independent guest means

Variant A: train_regressor_minilm_guestnorm_v1 (guest normalization, residual target)
•  Target during training/CV:
•  y_resid = y_log − mean_train(y_log | guest)
•  So the head learns only the part of performance not explained by the guest’s overall baseline
•  Features:
•  Only the text embedding
•  Inference scoring:
•  score_resid = regressor(embedding)
•  final_score = score_resid + guest_mean_log_views(guest)  (fallback to global mean if guest unknown)
•  What it’s doing conceptually:
•  Deconfounds “guest popularity” from the signal by removing the per-guest average first, then adds it back at inference
•  Pros:
•  Robust on small datasets; focuses the model on content differences within a guest
•  Less risk of the model over-relying on the guest identity
•  Cons:
•  If guest identity genuinely interacts with content in complex ways, this linear residual approach might miss some of that

Variant B: train_regressor_minilm_guestfeat_v1 (guest as a feature, target encoding)
•  Target during training/CV:
•  y_log (no residualization)
•  Features:
•  Concatenate the embedding with a 1-D guest feature = mean_train(y_log | guest)
•  Inference scoring:
•  guest_feature = guest_mean_log_views(guest) (fallback to global mean if unknown)
•  final_score = regressor([embedding; guest_feature])
•  What it’s doing conceptually:
•  Lets the regressor learn how to combine text and a guest “popularity prior” directly
•  Pros:
•  Can model linear interactions between content and guest prior in one step
•  Cons:
•  With small N, the model may over-rely on the guest feature and underuse the text
•  Slightly more fragile if guest priors are noisy

Leakage control (both variants)
•  For each CV fold:
•  Compute guest means only on the training split
•  Apply those means to build residuals (guestnorm) or features (guestfeat) for the validation split
•  Unknown guest in a fold (or at inference) falls back to the global mean

Artifacts and metadata differences
•  Guest normalization pickle includes:
•  target: "y_log_residual"
•  guest_means and guest_global_mean; scorer must add the mean back at inference
•  Guest feature pickle includes:
•  features: "embedding + guest_mean_log_views", target: "y_log"
•  guest_means and guest_global_mean; scorer must append this feature at inference

When to choose which
•  Prefer guestnorm if:
•  You want the head to focus on content and to generalize across/within guests more evenly
•  Dataset is small and you want to reduce the chance of the model latching onto guest identity
•  Consider guestfeat if:
•  You expect the guest baseline to be informative and want the head to blend it with text
•  CV shows a clear Spearman improvement over guestnorm
•  If metrics are similar, default to guestnorm for robustness and interpretability

Unknown/rare guests
•  Both variants fall back to a global mean when the guest isn’t present in the training stats
•  We can add optional smoothing (e.g., shrink per-guest mean toward global based on sample count) if needed

How this plugs into inference
•  Guest normalization scorer:
•  predict residual then add guest mean baseline
•  Guest feature scorer:
•  append guest mean feature to the embedding then predict directly
•  Both paths are straightforward to implement alongside your existing inference script; we’d add a --guest flag and choose the right scorer class based on the model you pass in

Bottom line
•  Train and compare both with your v4 dataset (reports already generated in their run folders)
•  If one shows clearly higher CV Spearman, use that; otherwise, guestnorm is the safer default for now