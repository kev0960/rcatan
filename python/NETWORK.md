# History of network improvements

One subsection per training run, newest first in each section. For each run:
what it tested, what changed from the run before, and how it came out.

## Transformer based

Actor head `--actor-head transformer` (`train/network_transformer.py`): the
graph encoder turns the state into node, road and global tokens, a board
transformer mixes them, and 300 learned action tokens (one per action) read
the board through cross-attention. Each action's logit comes from its own
token. Nothing tells an action token which part of the board it is about; it
has to learn where to look.

Common to all transformer runs:

- GAE with lambda 0.95, discount 0.996.
- Graph encoder with 3 layers, `GraphCritic` as the critic (the same as the
  entity runs).
- 128 games x 3000 steps per iteration, 2 actor passes, 1 critic pass.
- Linear learning-rate decay from 1e-4 to 3e-5 over 600 iterations.
- Baseline: 20261102-230211, except 194817 (20261107-155220, played 2 vs 2).

### 20261107-194817: wired action tokens (`transformer-wired`)

- **Tests:** whether the transformer catches up with the graph network when
  each action token doesn't have to learn where on the board it acts.
- **Changes from 075928:**
  - `--actor-head transformer-wired`: each action token starts as its learned
    vector plus the board tokens it acts on (a fixed 300 x 127 source
    matrix). A road gets its road token and its two end nodes, a settlement
    or city its node, a robber move its tile's six corners, and everything
    else the global token. The attention still reads the whole board.
  - The wired `action_emb` starts at std 1.0.
  - The state has the card estimates (2031 floats), as in 155220.
  - Baseline: the fully trained 155220, the best graph run, played 2 vs 2
    (parity 50%) instead of 1 vs 3.
  - Minibatch 4096 (2 micro-batches of 2048), gradient clipping at 2.0.
- **Result:**
  - vs baseline (2 vs 2): 9.0% average over iterations 1-50, 30.0% over
    101-150, 40.8% over 201-250, 49.3% over 401-450 and 52.8% over 501-600.
    It flattened only in the last 100 iterations. The earlier transformers
    flattened around iteration 400.
  - Entropy fell steadily: 0.36, 0.32, 0.28, 0.26 and 0.24 by 100-iteration
    block, lower than the graph runs. KL went from 0.006 to 0.003 as the
    learning rate decayed. Explained variance was about 0.63.
  - Stopped by accident at iteration 160 and resumed with no break in the
    curves.
  - About 8 h 15 min, against 3 h 19 min for 155220 (about 2.5x slower).
- **Head-to-head vs 155220** (`train.compare`, 5000 games per matchup):
  - 1 194817 vs 3 155220: 24.3% ± 1.2% (parity 25%).
  - 1 155220 vs 3 194817: 24.5% ± 1.2% (parity 25%).
  - 2 vs 2: 50.0% ± 1.4% (parity 50%).
  - Unlike every earlier pair, neither policy loses when it is alone at the
    table.
- **Verdict:** the first transformer to match the best graph run, but no
  better, at 2.5x the training time. `entity-robber` stays the default. Two
  different architectures ending at the same strength suggest the training
  setup (self-play, reward, 600 iterations), not the network, is now the
  limit.

### 20261106-075928: larger pos_emb init, looser gradient clipping

- **Tests:** whether the transformer learns faster per sample when its
  board tokens are easy to tell apart, while still learning (not
  hand-wiring) which board token each action should read.
- **Changes from 222120:**
  - `pos_emb` starts at std 1.0 instead of 0.02. The node tokens come out of a
    LayerNorm at about std 1, so at 0.02 the position was about 2% of each
    token, and empty nodes or roads looked almost the same to the attention.
  - `--max-grad-norm` 2.0 instead of 0.5. At 0.5 every step from iteration
    51 on was clipped, so the clip no longer singled out outliers.
- **Watch:** vs baseline before iteration 100 (222120: 2.9%, 115058: 8.0%),
  `actor_grad_clipped` well below 100%, and entropy not dropping below
  115058's.

### 20261105-222120: smaller transformer, 4096 minibatch, gradient clipping

- **Tests:** whether a cheaper, more stable transformer can match the
  entity head (115058).
- **Changes from 230947:**
  - Action reader is cross-attention only: no 300 x 300 self-attention among
    action tokens.
  - FFN width 2x hidden (256) instead of 4x (512).
  - The two transformers run in bfloat16 autocast on the GPU. The graph
    encoder and the readout stay in float32.
  - 4096 minibatch through gradient accumulation (2 micro-batches of 2048),
    so about 188 actor steps per iteration instead of about 750.
  - Gradient-norm clipping at 0.5, with the norms logged.
  - 0.82M actor parameters instead of 1.22M (`GraphActor`: 0.32M).
- **Result:**
  - No KL spikes (max 0.0085). Clipping capped several outlier steps: the
    largest was 35x the iteration's mean norm (iteration 41), others 16-17x
    (iterations 234, 537).
  - The mean actor gradient norm grew from 0.05 to about 0.95, so from
    iteration 51 every step was clipped.
  - vs baseline: 16.3% average over iterations 501-600, against 21.7% for 115058. It caught up early (2.9% vs 8.0% at iterations 1-100) and then
    flattened after about iteration 400.
  - Entropy ended lower than 115058's (0.23 vs 0.29) despite playing worse:
    it committed early rather than running out of capacity.
  - 7 h 51 min, against 3 h 35 min for 115058.
- **Verdict:** stable, but still behind the entity head. Led to 075928.

### 20261105-210942: first 4096-minibatch run (stopped at iteration 81)

- **Tests:** the smaller transformer with a 4096 minibatch, before gradient
  clipping existed.
- **Result:** a KL spike of 0.050 at iteration 46 (usual: 0.005-0.007). This
  run is why clipping was added. Stopped and restarted as 222120.

### 20261104-230947: first transformer (stopped at iteration 534)

- **Tests:** whether learned action tokens with attention over the whole
  board beat the entity head's per-action readouts.
- **Setup:**
  - A full `nn.TransformerDecoder`, with self-attention among the action
    tokens and a 512-wide FFN.
  - Minibatch 1024, no gradient clipping.
- **Result:**
  - About 6x slower per iteration than 115058.
  - 14 KL spikes above 0.015 (up to 0.064).
  - vs baseline: 11.0% at the end, against 7.5-22% for 115058 over the same
    iterations.
- **Verdict:** too slow and unstable. Its checkpoints no longer load into the
  current `TransformerActor`.

## Graph based

`--network graph` (`train/network_graph.py`): a 3-layer graph encoder
(`TransformerConv` over the 54 nodes and 72 roads). The default "entity"
actor head scores roads from their two end nodes, settlements and cities from
their node, and the other actions from a summary of the board. The actor and
critic are separate networks.

### 20261107-155220: bayesian estimation

Include the bayesian estimation of the other player's card from the current
player's perspective.

```
~/learn/rcatan/python on  main [!] took 56s
❯  uv run --no-sync python -m train.compare 20261107-155220 20261102-230211 --matches 3000
A: 20261107-155220 (graph)  [checkpoints/20261107-155220/actor_latest.pt]
B: 20261102-230211 (graph)  [checkpoints/20261102-230211/actor_latest.pt]
3000 games per matchup

1 A vs 3 B: win  22.9% ± 1.5% (parity 25%)  | vp/seat 6.83 vs 7.14  | steps 221
1 B vs 3 A: win  16.0% ± 1.3% (parity 25%)  | vp/seat 6.68 vs 7.27  | steps 230
2 A vs 2 B: win  54.0% ± 1.8% (parity 50%)  | vp/seat 7.17 vs 6.99  | steps 226

~/learn/rcatan/python on  main [!] took 55s
❯  uv run --no-sync python -m train.compare 20261107-155220 20261106-221927 --matches 3000
A: 20261107-155220 (graph)  [checkpoints/20261107-155220/actor_latest.pt]
B: 20261106-221927 (graph)  [checkpoints/20261106-221927/actor_latest.pt]
3000 games per matchup

1 A vs 3 B: win  25.2% ± 1.6% (parity 25%)  | vp/seat 7.23 vs 7.21  | steps 230
1 B vs 3 A: win  22.0% ± 1.5% (parity 25%)  | vp/seat 7.03 vs 7.28  | steps 234
2 A vs 2 B: win  52.8% ± 1.8% (parity 50%)  | vp/seat 7.30 vs 7.16  | steps 233
```

Slightly better than previous runs 
  - 53% advantage over the robber head fix.
  - 54% advantage over the previous "best" graph.

### 20261106-221927: robber head (`entity-robber`)

- **Tests:** whether robber moves score better from the tile they target
  than from the pooled board summary. In the old entity head, all 19 robber
  moves came from the same pooled vector, so the head could not tell one tile
  from another except through 19 separate output weights.
- **Changes from 115058:**
  - `--actor-head entity-robber` (now the default): each robber move is
    scored from the mean and max of its tile's 6 corner nodes plus the global
    features (`robber_head`).
  - `--max-grad-norm` 2.0 (115058 had no clipping). Almost no steps were
    clipped (under 1%), so this should not have changed the outcome.
  - Built on a tree that already had the card estimate in the engine, but
    the estimate is not in the state yet, so the actor could not see it.
- **Result:**
  - vs baseline (fully trained 230211): 23.4% average over iterations
    501-600, the best of any run against this baseline (115058: 21.7%,
    transformer 075928: 18.5%). Ahead of 115058 in 5 of the 6
    100-iteration blocks (9.9% vs 8.0%, 15.8 vs 13.2, 17.7 vs 18.8,
    19.9 vs 17.7, 21.3 vs 19.3, 23.4 vs 21.7), and still rising at the end.
  - Games against random bots are shorter (154 steps vs 158).
  - Entropy ended lower (0.26 vs 0.29); KL, clip fraction and explained
    variance were close to 115058's.
  - 3 h 13 min.
- **Head-to-head vs 230211** (`train.compare`, 10000 games per matchup):
  - 1 221927 vs 3 230211: 22.5% ± 0.8% (parity 25%).
  - 1 230211 vs 3 221927: 18.5% ± 0.8% (parity 25%).
  - 2 vs 2: 53.3% ± 1.0% for 221927 (parity 50%).
  - Whichever policy is alone at the table loses, so the 1 vs 3 numbers
    include a penalty for being the odd one out. With both lone results
    compared, 221927 does 4 points better as the lone player, and it wins the
    2 vs 2. The validation "vs baseline" check is always 1 vs 3, so it
    understates this run.
- **Verdict:** the first run to beat 230211, by a small margin.

### 20261103-224739: flat actor head

- **Tests:** whether the per-action entity heads matter, or whether one flat
  layer over a pooled board summary is enough.
- **Changes from 115058:** the actor head only (`--actor-head flat`).
- **Result:** clearly behind 115058 in head-to-head play from iteration 130
  on. vs baseline: 13.6%, against 20.6% for 115058.
- **Verdict:** the entity head stays. The flat head was removed, so these
  checkpoints no longer load.

### 20261103-185950: global fusion (stopped at iteration 288)

- **Tests:** whether mixing the global features into every node inside the
  encoder (global fusion) helps.
- **Changes from 115058:** `global_fusion` only.
- **Result:** even with 115058 or slightly behind in head-to-head play at
  every checkpoint up to iteration 250. For example, at iteration 250: 22.5%
  for 1 fusion vs 3 non-fusion, and 26.9% the other way round. vs baseline:
  16.9% at iteration 280.
- **Verdict:** no gain. Global fusion was removed on 2026-11-05, and these
  checkpoints no longer load.

### 20261103-115058: critic lr 1x, lr floor 3e-5 (best entity run since the steal-card fix)

- **Tests:**
  - whether the critic learns better at the same learning rate as the actor
    (1x) than at the 10x used in 082357;
  - whether a higher minimum learning rate (3e-5 instead of 1e-5) keeps the
    policy improving late in the run.
- **Setup:**
  - The same as 230211 (128 games, 2 actor passes, 1 critic pass, linear
    decay), but with the minimum learning rate at 3e-5.
  - Trained after the steal-card fix (`7dea96b`).
- **Result:**
  - Even with 230211 in head-to-head play at every iteration compared up to 452.
  - vs baseline (fully trained 230211): 20.6% at the end, still rising
    (21.7% average over iterations 501-600).
  - One KL spike (0.016 at iteration 32).
- **Verdict:** the reference run for every later experiment (fusion, flat,
  transformer).

### 20261103-105716: critic lr 1x, constant learning rate (stopped at iteration 143)

- **Tests:** a critic learning rate of 1x with no learning-rate decay.
- **Result:** vs baseline 11.3% at iteration 140. Replaced by 115058, which
  adds the linear decay.

### 20261103-082357: critic lr 10x with linear decay (stopped at iteration 418)

- **Tests:** whether a 10x critic learning rate (critic 1e-3, actor 1e-4)
  gives better advantages.
- **Result:** vs baseline 17.9% at iteration 410. No clear gain over 1x, so
  the default went back to 1x (`--critic-lr-scale 1`).

### 20261103-080935: critic lr 10x, constant learning rate (stopped at iteration 37)

- **Tests:** the same as 082357, but started without `--lr-decay linear` by
  mistake. Stopped and restarted as 082357.

### 20261102-230211: larger batch, fewer passes, 600 iterations (current baseline)

- **Tests:** whether more games per update and fewer passes over each buffer
  give a stronger bot than 211405.
- **Changes from 211405:**
  - 128 games per iteration instead of 64.
  - 2 actor passes and 1 critic pass instead of 4 and 2.
  - 600 iterations with linear decay down to 1e-5.
- **Result:**
  - vs baseline (211405): 43.9%, by far the biggest jump in this history.
  - Most of the gain came in one step around iterations 400-470; it was flat
    after that.
  - Started the same minute the steal-card fix (`7dea96b`) was committed. Its
    `config.json` records `328754f` with uncommitted changes.
- **Verdict:** the baseline for every run since 2026-11-03.

### 20261102-211405: linear learning-rate decay

- **Tests:** whether lowering the learning rate over the run helps the
  policy settle.
- **Changes from 194612:** `--lr-decay linear`, 300 iterations.
- **Result:** vs baseline (194612): 26.3%, about equal; entropy dropped to
  0.36. Later extended by resuming at a fixed learning rate.
- **Verdict:** linear decay became the default for every later run.

### 20261101-215109: 6-layer actor

- **Tests:** whether a deeper actor encoder (6 layers, `--actor-layers 6`)
  plays better. The critic stays at 3 layers.
- **Result:**
  - vs baseline (194612, the 3-layer run): about 20% over iterations 510-600,
    clearly below equal.
  - It reached 20% by iteration 220 and stayed there.
  - Its updates were too large for the learning rate kept from the 3-layer
    setup.
- **Verdict:** stay at 3 layers.

### 20261101-194612: 2 critic passes (stopped at about iteration 365)

- **Tests:** whether fewer critic passes per buffer (2 instead of 4) stop the
  critic from overfitting each batch.
- **Result:**
  - The critic's `explained_var_mc` was higher than in 125508 at the same
    iterations.
  - About 20 s per iteration instead of 25 s.
  - vs baseline (FF run 185907): 32.0%, and 36-39% for 1 graph vs 3 FF in
    head-to-head play.
- **Verdict:** 2 critic passes became the default. Baseline for 215109 and 211405.

### 20261101-125508: first graph run with GAE

- **Tests:** the graph network with GAE (lambda 0.95) against the
  fully trained FF GAE bot (123034).
- **Setup:** 4 critic passes, 300 iterations.
- **Result:**
  - vs baseline (123034): 34.7%, so the graph network beats FF at the same
    number of iterations.
  - The critic stayed well above 0 explained variance. The Monte Carlo graph
    run (225020) had gone negative by about iteration 20.

### 20261031-225020: first graph run (Monte Carlo returns)

- **Tests:** the graph network against FF on the 2016-float state.
- **Result:**
  - Beat the FF run 092834 in head-to-head play: 35.2% for 1 graph vs 3 FF,
    17.0% for 1 FF vs 3 graph (500 games each).
  - The critic memorized each batch: training loss 0.046 against a return
    variance of 0.59, and negative explained variance on new data.
- **Verdict:** the graph encoder is better, but the critic needed GAE and
  fewer passes.

## Feed forward network

`--network ff` (`train/network_ff.py`): an MLP over the flat 2016-float
state.

### 20261102-203517: potential-based reward shaping (PBRS)

- **Tests:** replacing the +0.1 per VP reward with potential-based VP shaping
  (the winner gets +1; the shaping cancels out over a game).
- **Result:** lost to 123034, the same setup with the old reward, at the same
  300 iterations: 19.9% (25% = equal).
- **Verdict:** PBRS was reverted.

### 20261101-185907: GAE, 600 iterations (best FF run)

- **Tests:** whether the FF network keeps improving with twice the
  iterations.
- **Result:** the strongest FF bot. It was the baseline for 194612, which
  beat it at 32-39%. It predates the steal-card fix and was never compared
  directly with 123034.

### 20261101-123034: GAE

- **Tests:** GAE (lambda 0.95) instead of Monte Carlo returns, for a full
  300 iterations.
- **Result:**
  - The critic predicts outcomes better than with Monte Carlo returns.
  - The policy levelled off after about 150 iterations.
  - Baseline for 125508 and the PBRS run 203517.

### 20261101-120905: first GAE run (stopped at iteration 254)

- **Tests:** the first try of GAE on FF. Restarted as 123034.

### 20261031-092834: Monte Carlo returns

- **Tests:** a second FF run on the 2016-float state.
- **Result:** the critic was weak (explained variance about 0 by iteration
  100). It was the FF side of the first graph vs FF comparison (225020).

### 20261031-083317: first baseline on the 2016-float state

- **Tests:** retraining FF after the state gained the board information:
  dice numbers per node, ports, cards received per dice number and resource,
  public VP, longest road and dev-card counts.
- **Result:**
  - 99% against random bots, 248 steps per game.
  - 36.5%, 39% and 29% against its own checkpoints at 50%, 75% and 90% of the
    run, so it was still improving at iteration 300.
  - Entropy about 0.57.

### 20261030-221401: old 1002-float state

- **Tests:** the first checkpointed FF run, on the original 1002-float state
  (no dice numbers, ports or opponent information).
- **Result:** it beat random bots, but its checkpoints no longer load since
  the state changed size.
