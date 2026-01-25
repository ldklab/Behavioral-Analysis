import math
import multiprocessing
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH
from joblib import Parallel, delayed
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_JOBS = multiprocessing.cpu_count()

# Output folders
OUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(os.getcwd(), "aidev_results"))
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

print("Output folder:", OUT_DIR)

df_pr = load_dataset("hao-li/AIDev", "pull_request", split="train").to_pandas()
df_det = load_dataset("hao-li/AIDev", "pr_commit_details", split="train").to_pandas()
df_repo = load_dataset("hao-li/AIDev", "repository", split="train").to_pandas()

# Keep only columns we use
df_pr = df_pr[[
    "id", "number", "title", "body", "agent", "user_id", "user", "state",
    "created_at", "closed_at", "merged_at", "repo_id", "repo_url", "html_url"
]].copy()

df_repo = df_repo.rename(columns={"id": "repo_id"})[
    ["repo_id", "language", "stars", "forks", "license", "full_name"]
].copy()

df_pr = df_pr.merge(df_repo, on="repo_id", how="left")

# Patch-available PRs: only PR ids that exist in pr_commit_details
pr_with_patch = set(df_det["pr_id"].unique())
df_pr_patch = df_pr[df_pr["id"].isin(pr_with_patch)].copy()

top5_lang_patch = (
    df_pr_patch.dropna(subset=["language"])
    .groupby("language")["id"].nunique()
    .sort_values(ascending=False)
    .head(5)
)

LANGS_TO_RUN = top5_lang_patch.index.tolist()

# Agents to include
AGENTS_TO_KEEP = ["OpenAI_Codex", "Devin", "Cursor", "Claude_Code", "Copilot"]
_token_re = re.compile(r"[A-Za-z_]\w+|\d+|==|!=|<=|>=|->|=>|[{}()[\];,.<>+\-*/%=&|!:]")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Define metrics for visualization
METRICS = {
    "rep_5gram": "5-gram Repetition",
    "rep_3gram": "3-gram Repetition",
    "rep_tok": "Token Repetition",
    "entropy": "Shannon Entropy",
    # Weighted similarity metrics
    "sim_tfidf_add_vs_rem_w": "TF-IDF (Weighted)",
    "sim_jaccard_add_vs_rem_w": "Jaccard (Weighted)",
    "sim_fuzzy_add_vs_rem_w": "Fuzzy (Weighted)",
    # Unweighted similarity metrics
    "sim_tfidf_add_vs_rem": "TF-IDF (Unweighted)",
    "sim_jaccard_add_vs_rem": "Jaccard (Unweighted)",
    "sim_fuzzy_add_vs_rem": "Fuzzy (Unweighted)",
    # Reuse metrics
    "self_reuse_nn": "Self-Reuse (NN)",
    "cross_agent_reuse_nn": "Cross-Agent Reuse",
    "sim_repo_tfidf": "Repo-Reuse (TF-IDF)",
}


def safe_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def split_patch_to_added_removed(patch: str):
    if not isinstance(patch, str):
        return "", ""
    added, removed = [], []
    for line in patch.splitlines():
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])
    return "\n".join(added), "\n".join(removed)


def tokenize_code(text: str):
    text = safe_text(text)
    return _token_re.findall(text)


def ngrams(tokens, n: int):
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def repetition_ratio(tokens, n=None):
    if not tokens:
        return 0.0
    if n is None:
        total = len(tokens)
        uniq = len(set(tokens))
        return 1.0 - (uniq / total)
    ng = ngrams(tokens, n)
    if not ng:
        return 0.0
    total = len(ng)
    uniq = len(set(ng))
    return 1.0 - (uniq / total)


def shannon_entropy(tokens):
    if not tokens:
        return 0.0
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = len(tokens)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def jaccard_token(a: str, b: str):
    ta = set(tokenize_code(a))
    tb = set(tokenize_code(b))
    if not ta or not tb:
        return 0.0
    return float(len(ta & tb) / len(ta | tb))


def fuzzy_token_set_ratio(a: str, b: str, cap_tokens=800):
    """
    Fast similarity between added and removed code.
    Returns [0,1]. Uses RapidFuzz token_set_ratio over token strings.
    """
    A = tokenize_code(a)[:cap_tokens]
    B = tokenize_code(b)[:cap_tokens]
    if not A or not B:
        return 0.0
    sa = " ".join(A)
    sb = " ".join(B)
    return float(fuzz.token_set_ratio(sa, sb) / 100.0)


def fit_tfidf_vectorizer(texts, max_features=8000):
    vec = TfidfVectorizer(
        tokenizer=tokenize_code,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        max_features=max_features
    )
    X = vec.fit_transform(texts)
    X = normalize(X, norm="l2", axis=1)  # makes cosine = dot product
    return vec, X


def cosine_from_normalized_rows(A, B):
    """
    A and B are row-aligned normalized sparse matrices.
    cosine similarity = rowwise dot product.
    """
    # element-wise multiply then row-sum
    return (A.multiply(B)).sum(axis=1).A1


def make_minhash(tokens, num_perm=128, shingle_n=5):
    m = MinHash(num_perm=num_perm)

    if not tokens:
        m.update(b"__EMPTY__")
        return m

    if len(tokens) < shingle_n:
        for t in tokens[:50]:
            m.update(t.encode("utf-8"))
        return m

    for ng in ngrams(tokens, shingle_n):
        m.update((" ".join(ng)).encode("utf-8"))

    return m


def run_language_experiment(
        LANG: str,
        agents_keep=None,
        min_tokens=50,
        max_tokens_quantile=0.99,
        balance=False,
        repo_baseline_max_prs=10,
        lsh_threshold=0.5,
        shingle_n=5,
        num_perm=128,
        tfidf_max_features=8000,
):
    if agents_keep is None:
        agents_keep = AGENTS_TO_KEEP

    # Filter PRs (language + agent)
    df_prL = df_pr[df_pr["language"] == LANG].copy()
    df_prL["agent"] = df_prL["agent"].astype(str)
    df_prL = df_prL[df_prL["agent"].isin(agents_keep)].copy()

    if len(df_prL) == 0:
        return None

    # Patch rows for those PRs
    pr_ids = set(df_prL["id"].unique().tolist())
    df_detL = df_det[df_det["pr_id"].isin(pr_ids)].copy()

    keep_cols = ["pr_id", "sha", "filename", "status", "additions", "deletions", "changes", "patch"]
    keep_cols = [c for c in keep_cols if c in df_detL.columns]
    df_detL = df_detL[keep_cols].copy()

    if len(df_detL) == 0:
        return None

    # Split patch into added/removed per file
    df_detL["patch"] = df_detL["patch"].fillna("")
    patches = df_detL["patch"].tolist()

    split_results = Parallel(n_jobs=N_JOBS, prefer="processes")(
        delayed(split_patch_to_added_removed)(p) for p in tqdm(patches, desc="Splitting patches")
    )
    df_detL["add_f"] = [r[0] for r in split_results]
    df_detL["rem_f"] = [r[1] for r in split_results]

    # ----------------------------
    # File-level similarities
    # ----------------------------
    # Fit TF-IDF once on the corpus of file-level add/rem texts
    tfidf_corpus = pd.concat([df_detL["add_f"], df_detL["rem_f"]], ignore_index=True).fillna("").tolist()
    _, X_all = fit_tfidf_vectorizer(tfidf_corpus, max_features=tfidf_max_features)

    X_add = X_all[:len(df_detL)]
    X_rem = X_all[len(df_detL):]
    df_detL["sim_tfidf_file"] = cosine_from_normalized_rows(X_add, X_rem)

    # Jaccard computation
    add_list = df_detL["add_f"].tolist()
    rem_list = df_detL["rem_f"].tolist()

    jaccard_results = Parallel(n_jobs=N_JOBS, prefer="processes")(
        delayed(jaccard_token)(a, r) for a, r in tqdm(zip(add_list, rem_list), total=len(add_list), desc="Jaccard")
    )
    df_detL["sim_jacc_file"] = jaccard_results

    # Parallel Fuzzy computation
    fuzzy_results = Parallel(n_jobs=N_JOBS, prefer="processes")(
        delayed(fuzzy_token_set_ratio)(a, r) for a, r in
        tqdm(zip(add_list, rem_list), total=len(add_list), desc="Fuzzy")
    )
    df_detL["sim_fuzzy_file"] = fuzzy_results

    # weights based on added tokens
    df_detL["add_tok_count_f"] = df_detL["add_f"].apply(lambda x: len(tokenize_code(x)))
    df_detL["weight_f"] = df_detL["add_tok_count_f"].clip(lower=1)

    def weighted_mean(values, weights):
        v = np.asarray(values, dtype=float)
        w = np.asarray(weights, dtype=float)
        s = w.sum()
        if len(v) == 0 or s == 0:
            return 0.0
        return float(np.sum(v * w) / s)

    # Compute both weighted and unweighted (simple mean) aggregations
    pr_file_agg = (
        df_detL.groupby("pr_id", sort=False)
        .agg(
            # Weighted means (by token count)
            sim_tfidf_add_vs_rem_w=("sim_tfidf_file", lambda x: weighted_mean(x, df_detL.loc[x.index, "weight_f"])),
            sim_jaccard_add_vs_rem_w=("sim_jacc_file", lambda x: weighted_mean(x, df_detL.loc[x.index, "weight_f"])),
            sim_fuzzy_add_vs_rem_w=("sim_fuzzy_file", lambda x: weighted_mean(x, df_detL.loc[x.index, "weight_f"])),
            # Unweighted means (simple average across files)
            sim_tfidf_add_vs_rem=("sim_tfidf_file", "mean"),
            sim_jaccard_add_vs_rem=("sim_jacc_file", "mean"),
            sim_fuzzy_add_vs_rem=("sim_fuzzy_file", "mean"),
            # File count
            files_changed=("filename", "nunique") if "filename" in df_detL.columns else ("pr_id", "size"),
        )
        .reset_index()
    )

    # ----------------------------
    # Agent-level file metrics (aggregate directly from file-level data)
    # ----------------------------
    # Add agent info to file-level data
    pr_to_agent = dict(zip(df_prL["id"], df_prL["agent"]))
    df_detL["agent"] = df_detL["pr_id"].map(pr_to_agent)

    # Compute agent-level aggregations (both weighted and unweighted)
    agent_file_agg = (
        df_detL.groupby("agent", sort=False)
        .agg(
            # Weighted means (by token count) - agent level
            agent_sim_tfidf_w=("sim_tfidf_file", lambda x: weighted_mean(x, df_detL.loc[x.index, "weight_f"])),
            agent_sim_jaccard_w=("sim_jacc_file", lambda x: weighted_mean(x, df_detL.loc[x.index, "weight_f"])),
            agent_sim_fuzzy_w=("sim_fuzzy_file", lambda x: weighted_mean(x, df_detL.loc[x.index, "weight_f"])),
            # Unweighted means - agent level
            agent_sim_tfidf=("sim_tfidf_file", "mean"),
            agent_sim_jaccard=("sim_jacc_file", "mean"),
            agent_sim_fuzzy=("sim_fuzzy_file", "mean"),
            # Counts
            agent_total_files=("pr_id", "count"),
            agent_total_prs=("pr_id", "nunique"),
        )
        .reset_index()
    )

    # Save agent-level file metrics
    agent_agg_path = os.path.join(TAB_DIR, f"agent_file_metrics_{LANG}.csv")
    agent_file_agg.to_csv(agent_agg_path, index=False)

    # PR-level concatenated added/removed
    pr_added = (
        df_detL.groupby("pr_id", sort=False)["add_f"]
        .apply(lambda xs: "\n".join([safe_text(x) for x in xs if safe_text(x).strip()]))
        .reset_index(name="added_code")
    )

    pr_removed = (
        df_detL.groupby("pr_id", sort=False)["rem_f"]
        .apply(lambda xs: "\n".join([safe_text(x) for x in xs if safe_text(x).strip()]))
        .reset_index(name="removed_code")
    )

    # merge into PR table
    df_prL = df_prL.rename(columns={"id": "pr_id"})
    df_prL = (
        df_prL
        .merge(pr_added, on="pr_id", how="left")
        .merge(pr_removed, on="pr_id", how="left")
        .merge(pr_file_agg, on="pr_id", how="left")
    )

    df_prL["added_code"] = df_prL["added_code"].fillna("")
    df_prL["removed_code"] = df_prL["removed_code"].fillna("")

    # ----------------------------
    # Token counts + filters
    # ----------------------------
    df_prL["tok_count_add"] = df_prL["added_code"].apply(lambda x: len(tokenize_code(x)))

    df_prL = df_prL[df_prL["tok_count_add"] >= min_tokens].copy()
    if len(df_prL) == 0:
        return None

    cap_thr = df_prL["tok_count_add"].quantile(max_tokens_quantile)
    df_prL = df_prL[df_prL["tok_count_add"] <= cap_thr].copy()

    # Originality metrics (PR-level added code)
    added_codes = df_prL["added_code"].tolist()

    # tokenization
    tokens_list = Parallel(n_jobs=N_JOBS, prefer="processes")(
        delayed(tokenize_code)(code) for code in tqdm(added_codes, desc="Tokenizing")
    )
    df_prL["tokens_add"] = tokens_list

    # Compute all originality metrics in one parallel pass
    def compute_originality(tokens):
        return (
            repetition_ratio(tokens, n=None),
            repetition_ratio(tokens, n=3),
            repetition_ratio(tokens, n=5),
            shannon_entropy(tokens)
        )

    orig_results = Parallel(n_jobs=N_JOBS, prefer="processes")(
        delayed(compute_originality)(t) for t in tqdm(tokens_list, desc="Originality")
    )

    df_prL["rep_tok"] = [r[0] for r in orig_results]
    df_prL["rep_3gram"] = [r[1] for r in orig_results]
    df_prL["rep_5gram"] = [r[2] for r in orig_results]
    df_prL["entropy"] = [r[3] for r in orig_results]

    #  Self-reuse (within-agent) via MinHash NN
    df_prL["self_reuse_nn"] = 0.0

    for agent_name, g in df_prL.groupby("agent", sort=False):
        idxs = g.index.tolist()
        if len(idxs) < 2:
            continue

        # MinHash creation
        tokens_for_agent = [df_prL.at[i, "tokens_add"] for i in idxs]
        keys_for_agent = [str(df_prL.at[i, "pr_id"]) for i in idxs]

        mh_list = Parallel(n_jobs=N_JOBS, prefer="processes")(
            delayed(make_minhash)(t, num_perm, shingle_n) for t in
            tqdm(tokens_for_agent, desc=f"MinHash (self) [{agent_name}]")
        )

        # Build LSH index (must be sequential)
        lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
        minhashes = {}
        for key, mh in zip(keys_for_agent, mh_list):
            lsh.insert(key, mh)
            minhashes[key] = mh

        # Query nearest-neighbor
        for idx, (i, key, mh) in enumerate(zip(idxs, keys_for_agent, mh_list)):
            candidates = lsh.query(mh)

            best = 0.0
            for cand in candidates:
                if cand == key:
                    continue
                sim = mh.jaccard(minhashes[cand])
                if sim > best:
                    best = sim
            df_prL.at[i, "self_reuse_nn"] = float(best)

    # Repo-reuse TF-IDF baseline (FAST + time-aware + no leakage)
    df_prL["created_at_dt"] = pd.to_datetime(df_prL["created_at"], errors="coerce", utc=True)

    # Fit TF-IDF once for PR added_code
    pr_texts = df_prL["added_code"].fillna("").tolist()
    _, X_pr = fit_tfidf_vectorizer(pr_texts, max_features=tfidf_max_features)

    df_prL = df_prL.reset_index(drop=True)  # align rows with X_pr

    sim_repo = np.zeros(len(df_prL), dtype=float)

    for repo_id, grp in df_prL.groupby("repo_id", sort=False):
        grp_sorted = grp.sort_values("created_at_dt")
        idxs = grp_sorted.index.tolist()

        for pos, i in enumerate(idxs):
            # earlier PRs only
            prev = idxs[:pos]
            if not prev:
                continue

            # take last K earlier PRs (closest in time)
            prev = prev[-repo_baseline_max_prs:]

            # mean vector baseline
            baseline = X_pr[prev].mean(axis=0)
            # cosine = dot because normalized
            sim_repo[i] = float((X_pr[i].multiply(baseline)).sum())

    df_prL["sim_repo_tfidf"] = sim_repo

    # Size-stratified balancing
    if balance:
        df_prL["size_bin"] = pd.qcut(df_prL["tok_count_add"], q=4, duplicates="drop")

        parts = []
        for b, gb in df_prL.groupby("size_bin", observed=False, sort=False):
            counts = gb["agent"].value_counts()
            if any(a not in counts.index for a in agents_keep):
                continue

            n_min = int(counts.min())
            for a in agents_keep:
                parts.append(gb[gb["agent"] == a].sample(n=n_min, random_state=SEED))

        if parts:
            df_bal = pd.concat(parts, ignore_index=True)
            print(df_bal["agent"].value_counts())
        else:
            df_bal = df_prL.copy()
    else:
        df_bal = df_prL.copy()

    if "size_bin" in df_bal.columns:
        df_bal["size_bin"] = df_bal["size_bin"].astype(str)

    return df_bal


def generate_all_visualizations(results_by_lang):
    # Combine all languages
    df_all = pd.concat([
        df.assign(language=lang) for lang, df in results_by_lang.items()
    ], ignore_index=True)

    # Create output directories
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)

    # Repetitiveness Comparison (Box Plot)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    rep_metrics = ["rep_tok", "rep_3gram", "rep_5gram"]
    aliases = {
        'rep_tok': 'Unigram Repetition',
        'rep_3gram': '3-gram Repetition',
        'rep_5gram': '5-gram Repetition',
    }

    for ax, metric in zip(axes, rep_metrics):
        if metric not in df_all.columns:
            ax.set_visible(False)
            continue
        order = df_all.groupby("agent")[metric].median().sort_values(ascending=False).index
        sns.boxplot(data=df_all, x="agent", y=metric, order=order, ax=ax)
        ax.set_title(METRICS.get(metric, metric), fontsize=12, fontweight="bold")
        ax.set_xlabel("Agent")
        ax.set_ylabel(aliases.get(metric, metric), fontsize=12, fontweight="bold")
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "repetitiveness_boxplot.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    # Edit vs Rewrite Behavior (Violin Plot)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sim_metrics = ["sim_tfidf_add_vs_rem_w", "sim_jaccard_add_vs_rem_w", "sim_fuzzy_add_vs_rem_w"]
    aliases = {
        'sim_tfidf_add_vs_rem_w': 'TF-IDF Similarity',
        'sim_jaccard_add_vs_rem_w': 'Jaccard Similarity',
        'sim_fuzzy_add_vs_rem_w': 'Fuzzy Similarity',
    }

    for ax, metric in zip(axes, sim_metrics):
        if metric not in df_all.columns:
            ax.set_visible(False)
            continue
        sns.violinplot(data=df_all, x="agent", y=metric, ax=ax, inner="box")
        ax.set_title(METRICS.get(metric, metric), fontsize=12, fontweight="bold")
        ax.set_xlabel("Agent")
        ax.set_ylabel(aliases.get(metric, metric), fontsize=12, fontweight="bold")
        ax.tick_params(axis='x', rotation=45)

    # plt.suptitle("Edit vs Rewrite Behavior (Add-Rem Similarity)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "edit_vs_rewrite_violin.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    # Entropy Distribution
    if "entropy" in df_all.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        order = df_all.groupby("agent")["entropy"].median().sort_values(ascending=False).index
        sns.violinplot(data=df_all, x="agent", y="entropy", order=order, ax=ax, inner="quartile")
        ax.set_title("Token Diversity (Shannon Entropy)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Agent")
        ax.set_ylabel("Entropy")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        path = os.path.join(FIG_DIR, "entropy_distribution.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()

    # Save combined dataset
    path = os.path.join(TAB_DIR, "all_results_combined.csv")
    cols_to_drop = ["added_code", "removed_code", "tokens_add", "body", "title"]
    df_export = df_all.drop(columns=[c for c in cols_to_drop if c in df_all.columns])
    df_export.to_csv(path, index=False)

    return df_export


if __name__ == '__main__':
    results_by_lang = {}

    for LANG in LANGS_TO_RUN:
        df_res = run_language_experiment(LANG=LANG)
        if df_res is not None and len(df_res) > 0:
            results_by_lang[LANG] = df_res

    df = generate_all_visualizations(results_by_lang)

    pd.set_option('display.max_columns', None)

    print(df.groupby("agent")[["sim_tfidf_add_vs_rem_w", "sim_jaccard_add_vs_rem_w", "sim_fuzzy_add_vs_rem_w"]].agg(['mean', 'median', 'std']))
    print(df.groupby("agent")[["rep_tok", "rep_3gram", "rep_5gram", "entropy"]].agg(['mean', 'median', 'std']))
