# PI0 Intervention Lift Report

## Scope

This report captures the code and experiment work completed for the PI0 intervention-lift plan:

- Added richer FAACT runtime diagnostics for alarm reasons, candidate quality, diversity, and rejection causes.
- Expanded risk-model training to support concatenated feature fields, threshold sweeps, calibration summaries, and reproducible run ranking.
- Added stricter intervention controls including cooldown hooks, per-episode budgets, diversity floors, and hybrid candidate search.
- Re-ran PI0 training and evaluation sweeps on fixed seeds.

## Best risk-model ranking

`failure_prediction_runs/pi0_ranked_models_by_f1.json` ranks the new training runs. The best overall model on held-out sample metrics was:

- `failure_prediction_runs/pi0_sweep_combo_decision_auroc`
- Features: `feat_action_prefix_flat_10 + feat_action_chunk_mean`
- Decision-only: `true`
- Test AUROC: `0.8166`
- Test AUPRC: `0.3997`
- Test F1: `0.4390`

This model became the main scorer for the follow-up PI0 intervention sweeps.

## Matched evaluation summary

`failure_prediction_runs/pi0_eval_combo_grid_summary.json` contains the first matched seed-200 sweep.

Key outcomes:

- Baseline on 20 matched episodes: `0/20`
- Monitor-only on the same 20 episodes: `0/20`
- Best intervention configs each reached `1/10`:
  - `pi0_eval_combo_hybrid_t050_n8_m002_cap2_s200`
  - `pi0_eval_combo_hybrid_t055_n8_m002_cap2_s200`
  - `pi0_eval_combo_hybrid_t060_n12_m002_cap2_s200`

Takeaways:

- `hybrid` candidate search clearly dominated pure `obs_noise` and pure `action_noise`.
- The runtime frequently found lower-risk candidates, with negative mean best-candidate deltas, but those safer chunks still did not reliably recover the task.
- The stricter rules reduced some pointless swaps, but they did not unlock multi-success recovery on the matched probe.

## Targeted follow-up summary

`failure_prediction_runs/pi0_eval_combo_targeted_summary.json` captures the second, tighter sweep around the best hybrid settings and the alternate combined-feature model selected by F1.

Outcome:

- None of the targeted follow-up configs beat the earlier `1/10` ceiling.
- All follow-up runs finished `0/10`, even though many still showed strong negative candidate-risk deltas.

Interpretation:

- The failure detector is learning a useful ranking signal.
- The intervention search is finding actions that score as lower risk under that signal.
- The remaining bottleneck is action usefulness, not just action scoring. In practice, PI0 is often choosing "less risky" alternatives that still do not re-enter a successful recovery trajectory.

## Current conclusion

The plan work is implemented and the sweep evidence is now reproducible in-repo. The result is a stronger PI0 intervention framework with better observability and model-selection tooling, but the online success lift is still capped at `1/10` on the current task/setup rather than clearly surpassing it.
