"""
need_state_graph.py

Stage 2.5: turns the need-state LABELS produced by cluster_basket_embeddings.py
into need-state GRAPHS — and persists the edges, which the pipeline previously
computed and threw away.

WHY THIS EXISTS
───────────────
cluster_basket_embeddings.cluster_basket_embeddings() builds the basket
mutual-kNN edge list, hands it to Leiden, and returns only
(basket_id, need_state_cluster). The `edges` DataFrame — every cross-need-state
connection in the data — goes out of scope and is garbage collected. Likewise
cluster_basket_embeddings_gmm() computes a full n_components-wide posterior per
basket and then keeps only .max(axis=1), discarding the rest of the
distribution. Both discarded objects describe how need-states relate to each
other. This module recovers them.

TWO KINDS OF EDGE, AND THEY ARE NOT INTERCHANGEABLE
───────────────────────────────────────────────────
  ADJACENCY (undirected)  build_need_state_adjacency()
      "Need-state A and B border each other in embedding space — baskets on
      their boundary are genuinely ambiguous between the two."
      Built by contracting the basket-level mutual-kNN graph down to
      need-state grain (a quotient graph).
      USE FOR: placing a basket, finding substitutable/nearby need-states,
      understanding which states are hard to separate.
      DO NOT USE FOR: predicting where a household goes next. Households do
      not move along similarity edges. This graph has no direction and no
      time in it at all.

  TRANSITION (directed)   build_need_state_transitions()
      "Of households observed in need-state A in one week, X% were in
      need-state B the following week."
      Built by sequencing each household's own weeks — which is possible only
      because basket_id encodes (household_number, year_week_number).
      USE FOR: journeys, next-best-action, migration.

A third, weaker signal is also recovered for cross-checking:
  OVERLAP                 build_gmm_overlap()
      Mean GMM posterior mass that baskets assigned to component i place on
      component j. A probabilistic analogue of ADJACENCY. If it disagrees
      badly with the quotient graph, distrust both.

WHAT IS DELIBERATELY NOT PROVIDED
─────────────────────────────────
Centroid-to-centroid distance between need-states. It is the obvious first
instinct and the weakest option here: Leiden communities are arbitrary-shaped,
so an elongated or non-convex community's centroid can land in a region of
64-dim space where no basket actually exists, making the distance between two
such centroids close to meaningless. The quotient graph measures the same
intuition (are these two states near each other?) using only points that
really exist.

MEMORY
──────
The need-state-grain outputs are tiny (n_need_states^2 at worst — tens of
thousands of rows). The BASKET-grain edge list is not: build_basket_knn_graph()
already warns it can reach 1B+ edges at full population scale. save_basket_
edges() therefore caps and warns rather than silently writing hundreds of GB.

USAGE
─────
    import need_state_graph as nsg

    edges    = build_basket_knn_graph(basket_gnn_embeddings)   # keep it this time
    clusters = run_leiden_on_basket_graph(edges)

    adj   = nsg.build_need_state_adjacency(edges, clusters)
    trans = nsg.build_need_state_transitions(clusters)

    # which need-states border need-state 5 (similarity, NOT movement)
    nsg.neighbouring_need_states(adj, 5)

    # where households in need-state 5 actually go next (movement)
    nsg.next_need_states(trans, 5)
    nsg.possible_journeys(trans, from_need_state=5, depth=3)

    # everything for ONE household: its observed path, and where it can go
    nsg.journeys_for_household(clusters, trans, household_number=4213)

pipeline_main.py calls build_and_save_all() as Stage 2.5, so all of the
above is available from the saved parquet files without re-running anything:

    adj   = pd.read_parquet("../data/output/need_state_adjacency.parquet")
    trans = pd.read_parquet("../data/output/need_state_transitions.parquet")
    clus  = pd.read_parquet("../data/output/basket_need_state_clusters.parquet")
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import config

OUTPUT_DIR = config.OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

ADJACENCY_PATH   = os.path.join(OUTPUT_DIR, "need_state_adjacency.parquet")
TRANSITIONS_PATH = os.path.join(OUTPUT_DIR, "need_state_transitions.parquet")
OVERLAP_PATH     = os.path.join(OUTPUT_DIR, "need_state_gmm_overlap.parquet")
# Stage 2.5's own drill-down artifact — NOT cluster_basket_embeddings' Stage 2a
# cache, which is now basket_knn_edges_k<K>_<mutual|onedir>.parquet.
#
# The two shared this exact filename until 2026-09-26, so a Stage 2.5 write
# would land on top of the Stage 2a cache while its .manifest.json sidecar
# stayed behind, still vouching for a file a different writer had replaced.
# It never fired in production only because BASKET_EDGE_WRITE_CAP refuses
# anything over 50M edges and the real graph has ~508M. Below the cap it
# would have. Keep these names distinct.
BASKET_EDGES_PATH = os.path.join(OUTPUT_DIR, "basket_knn_edges.parquet")

# Writing the raw basket-level edge list is opt-in and capped — see MEMORY above.
BASKET_EDGE_WRITE_CAP = config.BASKET_EDGE_WRITE_CAP

# Transitions below this many observed household-weeks are reported but flagged;
# a 2-period training window gives each household only a handful of weeks, so
# thin cells are expected and should not be over-read.
MIN_TRANSITION_SUPPORT = config.MIN_TRANSITION_SUPPORT


# ─────────────────────────────────────────────
# Parsing household + week back out of basket_id
# ─────────────────────────────────────────────

def split_basket_id(clusters: pd.DataFrame, basket_id_col: str = "basket_id") -> pd.DataFrame:
    """
    basket_id is built as  household_number || '_' || year_week_number
    (basket_store.py:117), so the household identity the rest of the pipeline
    discards is still recoverable here. This is the only reason a household-
    level journey is computable at all downstream of clustering.

    Adds: household_number (int64), year_week_number (int64).
    """
    out = clusters.copy()
    parts = out[basket_id_col].astype(str).str.rsplit("_", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError(
            f"{basket_id_col} does not look like 'household_week' — got e.g. "
            f"{out[basket_id_col].iloc[0]!r}. This function assumes the basket_id "
            f"format built by basket_store.build_baskets_table()."
        )
    out["household_number"] = pd.to_numeric(parts[0], errors="coerce").astype("Int64")
    out["year_week_number"] = pd.to_numeric(parts[1], errors="coerce").astype("Int64")

    bad = out["household_number"].isna() | out["year_week_number"].isna()
    if bad.any():
        print(f"  WARNING: {bad.sum():,} basket_ids could not be parsed into "
              f"household/week and are dropped from journey analysis.")
        out = out[~bad]

    return out.astype({"household_number": "int64", "year_week_number": "int64"})


def _add_week_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    year_week_number is year*100 + week, so plain subtraction misreads the
    year boundary: 202552 -> 202601 is a ONE week step but a numeric gap of 49.
    Dense-ranking the distinct week values actually present in the data gives a
    gap measure that is correct across year ends without needing to know
    whether a given retail year has 52 or 53 weeks.

    ASSUMPTION: the data covers a CONTIGUOUS span of weeks. That holds for the
    full-population export (millions of households mean every week in the
    window appears), but it means gaps are UNDERSTATED if a week is missing
    from the dataset entirely — e.g. a warehouse outage, or a heavily filtered
    extract. A week absent everywhere is invisible to the ranking, so a true
    2-week gap across it would be measured as 1. Check that the distinct week
    count matches the window length before trusting max_week_gap on a
    filtered extract.
    """
    weeks = np.sort(df["year_week_number"].unique())
    rank_of = {w: i for i, w in enumerate(weeks)}
    df = df.copy()
    df["week_rank"] = df["year_week_number"].map(rank_of).astype(np.int64)
    return df


# ─────────────────────────────────────────────
# 1. ADJACENCY — contract the basket kNN graph to need-state grain
# ─────────────────────────────────────────────

def build_need_state_adjacency(
    edges: pd.DataFrame,
    clusters: pd.DataFrame,
    cluster_col: str = "need_state_cluster",
    drop_self_loops: bool = True,
) -> pd.DataFrame:
    """
    Contracts the basket-level mutual-kNN graph (the `edges` DataFrame
    cluster_basket_embeddings.build_basket_knn_graph() returns) down to one
    node per need-state.

    Parameters
    ----------
    edges    : basket_a, basket_b, weight   (from build_basket_knn_graph)
    clusters : basket_id, <cluster_col>     (from run_leiden_on_basket_graph)

    Returns
    -------
    DataFrame: need_state_a, need_state_b, n_edges, weight_sum, lift, share_a, share_b

    `weight_sum` alone is misleading — a large need-state has more edges simply
    because it has more baskets, so raw totals rank pairs by size rather than
    by affinity. `lift` corrects for that against a configuration-model null
    (expected cross-weight given each need-state's total edge weight):

        expected(A,B) = deg(A) * deg(B) / (2 * total_weight)
        lift          = observed / expected

    lift > 1  = these two need-states touch MORE than their sizes alone predict
    lift < 1  = they touch less; they are well separated
    lift ≈ 1  = no relationship beyond size

    `share_a` is the fraction of need-state A's total edge weight that runs to
    B — the asymmetric read ("how much of A's boundary is with B"), which is
    usually the more actionable number for a small state adjacent to a big one.
    """
    lab = clusters.set_index("basket_id")[cluster_col]

    out = pd.DataFrame({
        "need_state_a": edges["basket_a"].map(lab),
        "need_state_b": edges["basket_b"].map(lab),
        "weight": edges["weight"].to_numpy(),
    })

    unlabelled = out["need_state_a"].isna() | out["need_state_b"].isna()
    if unlabelled.any():
        # Leiden only labels baskets that appear in the edge list, so this
        # should be empty for a matched pair of inputs — a non-zero count
        # means clusters/ edges came from different runs.
        print(f"  WARNING: {unlabelled.sum():,} edges have an endpoint with no "
              f"{cluster_col} label — dropping. Check `edges` and `clusters` came "
              f"from the SAME clustering run.")
        out = out[~unlabelled]

    # .map() against a label Series yields float64 whenever any lookup missed,
    # and that dtype survives the filter above. Cast back to int64 so this
    # artifact stays JOINABLE with build_need_state_transitions()' output — a
    # float/int mismatch between the two silently produces empty joins in any
    # downstream analysis that combines adjacency with journeys.
    out = out.astype({"need_state_a": "int64", "need_state_b": "int64"})

    # Undirected: canonicalize so (5,12) and (12,5) aggregate together.
    a = np.minimum(out["need_state_a"].to_numpy(), out["need_state_b"].to_numpy())
    b = np.maximum(out["need_state_a"].to_numpy(), out["need_state_b"].to_numpy())
    out["need_state_a"], out["need_state_b"] = a, b

    agg = (
        out.groupby(["need_state_a", "need_state_b"], as_index=False)
           .agg(n_edges=("weight", "size"), weight_sum=("weight", "sum"))
    )

    # Configuration-model null. deg(X) counts every edge incident to X,
    # including within-X edges, which is what makes it a degree/volume term.
    deg = (
        pd.concat([
            agg[["need_state_a", "weight_sum"]].rename(columns={"need_state_a": "ns"}),
            agg[["need_state_b", "weight_sum"]].rename(columns={"need_state_b": "ns"}),
        ])
        .groupby("ns")["weight_sum"].sum()
    )
    total_w = agg["weight_sum"].sum()

    deg_a = agg["need_state_a"].map(deg).to_numpy()
    deg_b = agg["need_state_b"].map(deg).to_numpy()
    expected = deg_a * deg_b / (2.0 * total_w)

    agg["lift"] = agg["weight_sum"] / np.maximum(expected, 1e-12)
    agg["share_a"] = agg["weight_sum"] / np.maximum(deg_a, 1e-12)
    agg["share_b"] = agg["weight_sum"] / np.maximum(deg_b, 1e-12)

    if drop_self_loops:
        agg = agg[agg["need_state_a"] != agg["need_state_b"]]

    agg = agg.sort_values("lift", ascending=False).reset_index(drop=True)
    print(f"Need-state adjacency: {len(agg):,} need-state pairs over "
          f"{deg.index.nunique()} need-states "
          f"({'self-loops dropped' if drop_self_loops else 'self-loops kept'})")
    return agg


# ─────────────────────────────────────────────
# 2. TRANSITIONS — the actual journey graph
# ─────────────────────────────────────────────

def build_need_state_transitions(
    clusters: pd.DataFrame,
    cluster_col: str = "need_state_cluster",
    max_week_gap: Optional[int] = 1,
    min_support: int = MIN_TRANSITION_SUPPORT,
) -> pd.DataFrame:
    """
    Builds the DIRECTED need-state graph by walking each household's own weeks
    in order. This is the only graph in this module that can answer "where does
    a household go next" — the adjacency graph cannot, at any resolution.

    Parameters
    ----------
    max_week_gap : 1    only count a transition between CONSECUTIVE weeks
                   None count a transition between consecutive OBSERVED baskets,
                        however many weeks apart (households do not shop every
                        week, so this yields far more transitions at the cost of
                        conflating a 1-week and a 6-week step)
    min_support  : transitions observed fewer times than this are kept but
                   flagged via the `low_support` column rather than silently
                   dropped — with a ~8 week window, thin cells are expected.

    Returns
    -------
    DataFrame: from_need_state, to_need_state, n_transitions, prob, lift,
               avg_week_gap, low_support

    `prob` is P(next = B | current = A) — row-normalized, so each from_need_state
    sums to 1. This is the number to read for journeys.
    `lift` compares that to B's unconditional share of all destinations, so it
    separates "everyone goes to B because B is huge" (lift ≈ 1) from "households
    in A go to B specifically" (lift > 1).
    """
    df = split_basket_id(clusters[["basket_id", cluster_col]].dropna(subset=[cluster_col]))
    df = _add_week_rank(df)

    n_hh = df["household_number"].nunique()
    weeks_per_hh = len(df) / max(n_hh, 1)
    print(f"  Journey input: {len(df):,} baskets, {n_hh:,} households, "
          f"{weeks_per_hh:.1f} observed weeks per household on average")
    if weeks_per_hh < 3:
        print(f"  NOTE: {weeks_per_hh:.1f} weeks per household is thin for transition "
              f"analysis. The training SQL window is ~2 periods (~8 weeks) and "
              f"households do not shop every week. Treat first-order transitions as "
              f"indicative and widen the window in "
              f"ns_household_tpnb_week_agg_train.sql before modelling sequences.")

    df = df.sort_values(["household_number", "week_rank"])
    g = df.groupby("household_number", sort=False)
    df["to_need_state"] = g[cluster_col].shift(-1)
    df["next_week_rank"] = g["week_rank"].shift(-1)
    df["week_gap"] = df["next_week_rank"] - df["week_rank"]

    steps = df.dropna(subset=["to_need_state", "week_gap"]).copy()
    if max_week_gap is not None:
        before = len(steps)
        steps = steps[steps["week_gap"] <= max_week_gap]
        print(f"  Kept {len(steps):,} of {before:,} household-to-household steps "
              f"at max_week_gap={max_week_gap}")

    if len(steps) == 0:
        print("  WARNING: no transitions found — returning empty. Try max_week_gap=None.")
        return pd.DataFrame(columns=["from_need_state", "to_need_state", "n_transitions",
                                     "prob", "lift", "avg_week_gap", "low_support"])

    steps = steps.rename(columns={cluster_col: "from_need_state"})
    steps["to_need_state"] = steps["to_need_state"].astype(np.int64)

    agg = (
        steps.groupby(["from_need_state", "to_need_state"], as_index=False)
             .agg(n_transitions=("week_gap", "size"), avg_week_gap=("week_gap", "mean"))
    )

    row_total = agg.groupby("from_need_state")["n_transitions"].transform("sum")
    agg["prob"] = agg["n_transitions"] / row_total

    # Unconditional destination share — the null a journey should beat.
    dest_share = agg.groupby("to_need_state")["n_transitions"].sum() / agg["n_transitions"].sum()
    agg["lift"] = agg["prob"] / agg["to_need_state"].map(dest_share).to_numpy()

    agg["low_support"] = agg["n_transitions"] < min_support

    agg = agg.sort_values(["from_need_state", "prob"], ascending=[True, False]).reset_index(drop=True)
    print(f"Need-state transitions: {len(agg):,} directed pairs, "
          f"{agg['n_transitions'].sum():,} household-week steps "
          f"({agg['low_support'].sum():,} pairs below min_support={min_support})")
    return agg


# ─────────────────────────────────────────────
# 3. JOURNEY QUERIES — what the transition graph is for
# ─────────────────────────────────────────────

def next_need_states(
    transitions: pd.DataFrame,
    from_need_state: int,
    top_n: int = 5,
    exclude_self: bool = False,
) -> pd.DataFrame:
    """One step. Where do households in `from_need_state` go the following week?

    exclude_self=True drops the stay-put row, which is usually the largest by
    far — households mostly repeat their own pattern week to week — and hides
    the more interesting movement underneath it.
    """
    rows = transitions[transitions["from_need_state"] == from_need_state]
    if exclude_self:
        rows = rows[rows["to_need_state"] != from_need_state]
    return rows.nlargest(top_n, "prob").reset_index(drop=True)


def possible_journeys(
    transitions: pd.DataFrame,
    from_need_state: int,
    depth: int = config.JOURNEY_DEPTH,
    beam: int = config.JOURNEY_BEAM,
    exclude_self_loops: bool = True,
    min_path_prob: float = 0.0,
) -> pd.DataFrame:
    """
    Multi-step journeys out of a need-state, via beam search over the directed
    transition graph.

    Parameters
    ----------
    depth : how many weeks ahead to project
    beam  : how many partial paths to carry forward at each step
    exclude_self_loops : skip "stayed in the same need-state" steps, so the
        result shows actual movement rather than N weeks of standing still
    min_path_prob : prune paths below this cumulative probability

    Returns
    -------
    DataFrame: path (tuple of need-states), path_prob, weakest_step, low_support_steps

    IMPORTANT — `path_prob` is a product of first-order conditional
    probabilities. It assumes the chain is Markov: that where a household goes
    next depends only on where it is now, not how it got there. That assumption
    is NOT tested anywhere in this pipeline, and with ~8 weeks of data it cannot
    be tested well. Read multi-step paths as plausible routes, not forecasts,
    and check `low_support_steps` before quoting any of them.
    """
    idx = {}
    for row in transitions.itertuples(index=False):
        idx.setdefault(row.from_need_state, []).append(
            (row.to_need_state, row.prob, row.low_support)
        )

    paths = [((from_need_state,), 1.0, 1.0, 0)]
    finished = []

    for _ in range(depth):
        expanded = []
        for path, prob, weakest, low_n in paths:
            for to_ns, step_p, low in idx.get(path[-1], []):
                if exclude_self_loops and to_ns == path[-1]:
                    continue
                new_prob = prob * step_p
                if new_prob < min_path_prob:
                    continue
                expanded.append((
                    path + (to_ns,), new_prob, min(weakest, step_p), low_n + int(low),
                ))
        if not expanded:
            break
        expanded.sort(key=lambda r: r[1], reverse=True)
        paths = expanded[:beam]
        finished.extend(paths)

    if not finished:
        print(f"  No journeys found out of need-state {from_need_state}.")
        return pd.DataFrame(columns=["path", "path_prob", "weakest_step", "low_support_steps"])

    out = pd.DataFrame(finished, columns=["path", "path_prob", "weakest_step", "low_support_steps"])
    out["n_steps"] = out["path"].map(len) - 1
    return out.sort_values("path_prob", ascending=False).reset_index(drop=True)


def household_history(
    clusters: pd.DataFrame,
    household_number: int,
    cluster_col: str = "need_state_cluster",
) -> pd.DataFrame:
    """
    The need-state path one household has ALREADY taken, week by week.

    This is the observed history, not a prediction — and it is the thing the
    rest of the pipeline cannot give you, because after clustering a household
    exists only as a prefix inside basket_id. Returns empty if the household
    has no baskets in `clusters`.

    Returns
    -------
    DataFrame: year_week_number, <cluster_col>, basket_id  (ordered oldest first)
    """
    df = split_basket_id(clusters)
    hh = df[df["household_number"] == household_number]
    if hh.empty:
        print(f"  Household {household_number} has no baskets in this cluster table.")
        return pd.DataFrame(columns=["year_week_number", cluster_col, "basket_id"])
    return (
        hh.sort_values("year_week_number")[["year_week_number", cluster_col, "basket_id"]]
          .reset_index(drop=True)
    )


def journeys_for_household(
    clusters: pd.DataFrame,
    transitions: pd.DataFrame,
    household_number: int,
    cluster_col: str = "need_state_cluster",
    depth: int = config.JOURNEY_DEPTH,
    beam: int = config.JOURNEY_BEAM,
    exclude_self_loops: bool = True,
) -> dict:
    """
    End-to-end answer to "where can THIS household go next": look up the
    household's most recent observed need-state, then project forward through
    the transition graph.

    Returns a dict with:
        household_number
        current_need_state   — from its most recent week
        as_of_week           — which week that was
        history              — its observed path so far (household_history)
        next_steps           — one-step destinations, ranked
        journeys             — multi-step paths (possible_journeys)

    Same Markov caveat as possible_journeys(): the projection assumes where a
    household goes next depends only on where it is now, not how it got there.
    `history` is included precisely so you can eyeball whether that assumption
    looks reasonable for the household in front of you.
    """
    hist = household_history(clusters, household_number, cluster_col=cluster_col)
    if hist.empty:
        return {
            "household_number": household_number, "current_need_state": None,
            "as_of_week": None, "history": hist,
            "next_steps": pd.DataFrame(), "journeys": pd.DataFrame(),
        }

    latest = hist.iloc[-1]
    current = int(latest[cluster_col])

    return {
        "household_number": household_number,
        "current_need_state": current,
        "as_of_week": int(latest["year_week_number"]),
        "history": hist,
        "next_steps": next_need_states(transitions, current, exclude_self=exclude_self_loops),
        "journeys": possible_journeys(
            transitions, current, depth=depth, beam=beam,
            exclude_self_loops=exclude_self_loops,
        ),
    }


def neighbouring_need_states(
    adjacency: pd.DataFrame,
    need_state: int,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Which need-states BORDER this one (adjacency, not journeys). Handles the
    undirected canonicalization so you get neighbours regardless of which
    column the need-state landed in.

    Use this to find substitutable / easily-confused need-states. Do not read
    it as movement.
    """
    hit_a = adjacency[adjacency["need_state_a"] == need_state].copy()
    hit_a["neighbour"] = hit_a["need_state_b"]
    hit_a["share_of_this"] = hit_a["share_a"]

    hit_b = adjacency[adjacency["need_state_b"] == need_state].copy()
    hit_b["neighbour"] = hit_b["need_state_a"]
    hit_b["share_of_this"] = hit_b["share_b"]

    cols = ["neighbour", "n_edges", "weight_sum", "lift", "share_of_this"]
    return (
        pd.concat([hit_a[cols], hit_b[cols]])
          .nlargest(top_n, "lift")
          .reset_index(drop=True)
    )


# ─────────────────────────────────────────────
# 4. GMM POSTERIOR OVERLAP — cross-check on adjacency
# ─────────────────────────────────────────────

def build_gmm_overlap(
    basket_gnn_embeddings: pd.DataFrame,
    gmm,
    batch_size: int = 200_000,
) -> pd.DataFrame:
    """
    Recovers the posterior distribution cluster_basket_embeddings_gmm()
    computes and discards (it keeps only .max(axis=1) as `gmm_confidence`).

    For every basket, predict_proba gives its membership across ALL components.
    Averaging those distributions within each assigned component yields an
    n_components x n_components matrix: row i is "where does the probability
    mass of baskets assigned to i actually sit". The diagonal is purity; the
    off-diagonal is bleed between need-states.

    Batched because predict_proba materializes (n_baskets x n_components)
    floats, which at full population scale is large enough to matter.

    Returns
    -------
    DataFrame: gmm_component_a, gmm_component_b, mean_posterior, n_baskets_a

    NOTE ON COLUMN NAMES: these are GMM COMPONENT ids, which are a DIFFERENT
    labelling from the Leiden `need_state_cluster` ids used by
    build_need_state_adjacency() and build_need_state_transitions(). GMM
    component 5 and Leiden community 5 are unrelated. The columns are named
    gmm_component_* rather than need_state_* specifically so that joining this
    table to the adjacency or transition tables fails loudly instead of
    silently producing nonsense. To relate the two labellings, join through
    basket_id via basket_need_state_clusters.parquet (which carries both), or
    read the Adjusted Rand Index from compare_leiden_gmm().
    """
    from sklearn.preprocessing import normalize

    X = normalize(np.stack(basket_gnn_embeddings["gnn_embedding"].values))
    n_comp = gmm.n_components

    sums = np.zeros((n_comp, n_comp), dtype=np.float64)
    counts = np.zeros(n_comp, dtype=np.int64)

    for start in range(0, len(X), batch_size):
        chunk = X[start:start + batch_size]
        probs = gmm.predict_proba(chunk)
        assigned = probs.argmax(axis=1)
        np.add.at(sums, assigned, probs)
        np.add.at(counts, assigned, 1)

    mean_post = sums / np.maximum(counts[:, None], 1)

    a, b = np.meshgrid(np.arange(n_comp), np.arange(n_comp), indexing="ij")
    out = pd.DataFrame({
        "gmm_component_a": a.ravel(),
        "gmm_component_b": b.ravel(),
        "mean_posterior": mean_post.ravel(),
        "n_baskets_a": counts[a.ravel()],
    })
    diag = float(np.mean(np.diag(mean_post)))
    print(f"GMM overlap matrix: {n_comp}x{n_comp}, mean diagonal (purity) = {diag:.3f}")
    if diag < 0.6:
        print(f"  NOTE: mean purity {diag:.3f} is low — components overlap heavily, "
              f"so GMM need-state boundaries are soft. Worth comparing against the "
              f"Leiden adjacency graph before treating either as crisp segments.")
    return out


# ─────────────────────────────────────────────
# 5. PERSISTENCE
# ─────────────────────────────────────────────

def save_basket_edges(edges: pd.DataFrame, path: str = BASKET_EDGES_PATH,
                       cap: int = BASKET_EDGE_WRITE_CAP) -> Optional[str]:
    """
    Writes the raw BASKET-level edge list — the thing that lets you find the
    specific neighbouring baskets behind any need-state adjacency.

    Capped on purpose: build_basket_knn_graph() warns this can reach 1B+ edges
    at full population scale, which is hundreds of GB as parquet. Above `cap`
    this writes nothing and tells you rather than filling the disk.
    """
    if len(edges) > cap:
        print(f"  REFUSING to write {len(edges):,} basket edges to {path} — above the "
              f"{cap:,} cap. This would be hundreds of GB. Either raise `cap` "
              f"deliberately, or work from the need-state-grain outputs (which are "
              f"tiny) and recompute basket-level neighbours on demand for the "
              f"specific need-states you care about.")
        return None
    edges.to_parquet(path, index=False)
    print(f"  Saved {path} ({len(edges):,} basket-level edges)")
    return path


def build_and_save_all(
    edges: pd.DataFrame,
    leiden_clusters: pd.DataFrame,
    basket_gnn_embeddings: Optional[pd.DataFrame] = None,
    gmm=None,
    max_week_gap: Optional[int] = config.TRANSITION_MAX_WEEK_GAP,
    write_basket_edges: bool = config.WRITE_BASKET_EDGES,
) -> dict:
    """
    One call for the whole stage — intended to run straight after Stage 2 in
    pipeline_main.py, while `edges` is still in scope.

    Returns a dict of the produced DataFrames; also writes each to parquet.
    """
    print("\n[Stage 2.5a] Need-state adjacency (contracting the basket kNN graph)...")
    adjacency = build_need_state_adjacency(edges, leiden_clusters)
    adjacency.to_parquet(ADJACENCY_PATH, index=False)
    print(f"  Saved {ADJACENCY_PATH}")

    print("\n[Stage 2.5b] Need-state transitions (household week sequences)...")
    transitions = build_need_state_transitions(leiden_clusters, max_week_gap=max_week_gap)
    transitions.to_parquet(TRANSITIONS_PATH, index=False)
    print(f"  Saved {TRANSITIONS_PATH}")

    result = {"adjacency": adjacency, "transitions": transitions}

    if gmm is not None and basket_gnn_embeddings is not None:
        print("\n[Stage 2.5c] GMM posterior overlap (cross-check on adjacency)...")
        overlap = build_gmm_overlap(basket_gnn_embeddings, gmm)
        overlap.to_parquet(OVERLAP_PATH, index=False)
        print(f"  Saved {OVERLAP_PATH}")
        result["overlap"] = overlap

    if write_basket_edges:
        print("\n[Stage 2.5d] Raw basket-level edges...")
        save_basket_edges(edges)

    return result
