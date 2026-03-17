# PI0 Locality Probe

Current best saved PI0 intervention score remains `1/10` on the matched seed-200 probe.

## New locality-gated probe

Run:

- `hybrid`
- `risk_threshold=0.55`
- `num_candidate_chunks=12`
- `switch_margin=0.02`
- `min_candidate_l2_to_baseline=1.0`
- `min_candidate_prefix_l2_to_baseline=0.35`
- `max_candidate_tail_l2_to_baseline=1.75`
- `local_search_prefix_steps=8`

Outcome:

- `0/10` success
- `0` accepted interventions
- `16` rejected intervention attempts
- rejection reason: `no_local_recovery_candidate`

Useful diagnostic:

- mean best-candidate risk delta: `-0.1532`
- mean best-candidate prefix L2: `4.0939`
- mean best-candidate tail L2: `12.7828`

Interpretation:

- The risk model still finds lower-risk alternatives.
- The near-term prefix is changing enough to matter.
- The tail is diverging far too much, which explains why earlier accepted swaps were often low-risk but not useful.
- The next tuning direction is to soften the tail cap without dropping the new prefix-change requirement.

## Follow-up

A second looser locality probe was started with a much larger tail cap, but the RunPod SSH session dropped before metrics could be captured.
