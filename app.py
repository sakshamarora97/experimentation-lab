# Elements of A/B Testing: Intuition Lab 
# -------------------------------------------------------------
# Changes vs previous:
# - Performance: vectorized two-sample tests; caching MC draws/quantiles; fast plotting (decimation)
# - UX: spinners during heavy computations
# - Empirical variability tab: quantile 0.10–0.90, compare 3 variance estimators (bootstrap, naive s^2/n, true MC)
# - Randomization tab: clearer copy + visuals; design-effect and cluster-means explained
# - CUPED: correct implementation with external μ_X option → real variance reduction
# -------------------------------------------------------------

from __future__ import annotations
import math
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from scipy import stats

# -------------------------------------------------------------
# Page config
# -------------------------------------------------------------
st.set_page_config(page_title="A/B Testing Intuition Lab", layout="wide")

# -------------------------------------------------------------
# Utilities & small helpers
# -------------------------------------------------------------

def set_seed(seed: int | None):
    """Set global RNG for any non-RNG code path (we mainly use local RNGs)."""
    if seed is not None:
        np.random.seed(int(seed))

def explain(title: str, body: str):
    """Place verbose teaching copy in an expander."""
    with st.expander(title, expanded=False):
        st.markdown(body)

def _params_key(d: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """Immutable, hashable key for caching."""
    return tuple(sorted(d.items()))

# -------------------------------------------------------------
# RNG-based population drawers (vectorized; prefer local RNG)
# -------------------------------------------------------------
def rng_draw_from_population(rng: np.random.Generator, name: str, params: dict, size) -> np.ndarray:
    n = name.lower()
    if n == "normal":
        return rng.normal(params.get("mu", 0.0), params.get("sigma", 1.0), size)
    if n == "bernoulli":
        return rng.binomial(1, params.get("p", 0.5), size).astype(float)
    if n == "poisson":
        return rng.poisson(params.get("lam", 1.0), size).astype(float)
    if n == "exponential":
        return rng.exponential(params.get("scale", 1.0), size)
    if n == "lognormal":
        return rng.lognormal(params.get("meanlog", 0.0), params.get("sdlog", 1.0), size)
    if n == "pareto":
        alpha = params.get("alpha", 3.0); xm = params.get("scale", 1.0)
        return xm * (1 + rng.pareto(alpha, size))
    if n == "beta":
        return rng.beta(params.get("a", 2.0), params.get("b", 5.0), size)
    if n == "logit-normal":
        z = rng.normal(params.get("mu", 0.0), params.get("sigma", 1.0), size)
        return 1/(1+np.exp(-z))
    raise ValueError(f"Unknown distribution: {name}")

def draw_from_population(name: str, params: dict, size) -> np.ndarray:
    """Back-compat wrapper (uses global RNG)."""
    return rng_draw_from_population(np.random.default_rng(), name, params, size)

def population_moments(name: str, params: dict) -> tuple[float | None, float | None]:
    name = name.lower()
    if name == "normal":
        return float(params.get("mu", 0.0)), float(params.get("sigma", 1.0) ** 2)
    if name == "bernoulli":
        p = params.get("p", 0.5); return float(p), float(p*(1-p))
    if name == "poisson":
        lam = params.get("lam", 1.0); return float(lam), float(lam)
    if name == "exponential":
        s = params.get("scale", 1.0); return float(s), float(s**2)
    if name == "lognormal":
        m = params.get("meanlog", 0.0); s = params.get("sdlog", 1.0)
        mu = np.exp(m + 0.5*s*s)
        var = (np.exp(s*s)-1)*np.exp(2*m + s*s)
        return float(mu), float(var)
    if name == "pareto":
        a = params.get("alpha", 3.0); xm = params.get("scale", 1.0)
        if a <= 1: return None, None
        mu = a*xm/(a-1)
        if a <= 2: return float(mu), None
        var = (a*xm**2)/((a-1)**2*(a-2))
        return float(mu), float(var)
    if name == "beta":
        a = params.get("a", 2.0); b = params.get("b", 5.0)
        mu = a/(a+b); var = (a*b)/(((a+b)**2)*(a+b+1))
        return float(mu), float(var)
    if name == "logit-normal":
        return None, None
    return None, None

def sample_means(name: str, params: dict, n: int, R: int) -> np.ndarray:
    draws = draw_from_population(name, params, size=(R, n))
    return draws.mean(axis=1)

# --- Two-sample tests (row-wise vectorized & single-run versions) ---
def two_sample_welch_ttest(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (t_stat, p_value) for a single Welch test."""
    res = stats.ttest_ind(x, y, equal_var=False)
    return float(res.statistic), float(res.pvalue)

def pooled_ttest(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (t_stat, p_value) for a single pooled-variance test."""
    res = stats.ttest_ind(x, y, equal_var=True)
    return float(res.statistic), float(res.pvalue)

def two_sample_welch_ttest_matrix(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Welch t-test across rows (replications)."""
    res = stats.ttest_ind(X, Y, axis=1, equal_var=False)
    return np.asarray(res.statistic), np.asarray(res.pvalue)

def pooled_ttest_matrix(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized pooled t-test across rows (replications)."""
    res = stats.ttest_ind(X, Y, axis=1, equal_var=True)
    return np.asarray(res.statistic), np.asarray(res.pvalue)

def normal_power_two_sample(delta: float, sigma: float, n: int, alpha: float, two_sided: bool = True) -> float:
    if sigma <= 0 or n <= 0: return np.nan
    se = math.sqrt(2*(sigma**2)/n)
    ncp = delta/se
    if two_sided:
        z = stats.norm.ppf(1 - alpha/2)
        return float(stats.norm.cdf(-z - ncp) + 1 - stats.norm.cdf(z - ncp))
    z = stats.norm.ppf(1 - alpha)
    return float(1 - stats.norm.cdf(z - ncp))

def solve_n_for_power(delta: float, sigma: float, alpha: float, target_power: float, two_sided: bool=True, n_max: int=200000) -> int:
    if delta <= 0 or sigma <= 0 or target_power <= 0: return 0
    lo, hi = 2, n_max
    for _ in range(40):
        mid = (lo+hi)//2
        p = normal_power_two_sample(delta, sigma, mid, alpha, two_sided)
        if p >= target_power: hi = mid
        else: lo = mid + 1
    return hi

def estimate_sigma_from_dist(name: str, params: dict, fallback_n: int = 200000) -> float:
    _, var = population_moments(name, params)
    if var is not None: return float(math.sqrt(var))
    x = draw_from_population(name, params, min(fallback_n, 200000))
    return float(np.std(x, ddof=1))

# -------------------------------------------------------------
# Cached helpers for heavy Monte-Carlo
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_mc_draws(dist: str, params_key: tuple, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    params = dict(params_key)
    return rng_draw_from_population(rng, dist, params, n)

@st.cache_data(show_spinner=False)
def cached_true_quantile(dist: str, params_key: tuple, q: float, seed: int, n_mc: int = 200_000) -> float:
    """Return population quantile (analytic when possible; else MC)."""
    params = dict(params_key)
    if dist.lower() == "beta":
        return float(stats.beta(params["a"], params["b"]).ppf(q))
    # fallback MC
    x = cached_mc_draws(dist, params_key, n_mc, seed)
    return float(np.quantile(x, q))

# -------------------------------------------------------------
# Header & global sidebar controls
# -------------------------------------------------------------
st.title("Interactive Simulations for A/B Testing Concepts")

st.sidebar.header("Global settings")
seed = st.sidebar.number_input("Random seed (optional)", value=123, step=1)
set_seed(seed)

R_default = st.sidebar.slider("Default replications R (experiments)", 10, 20000, 1000, step=10)
n_default = st.sidebar.slider("Default sample size n (per group where applicable)", 5, 100000, 200, step=5)
alpha_default = st.sidebar.select_slider("Significance level α", options=[0.01, 0.02, 0.025, 0.05, 0.10], value=0.05)
side_tail = st.sidebar.radio("Test tail", ("Two-sided", "One-sided"), index=0)
TWO_SIDED = (side_tail == "Two-sided")

st.sidebar.markdown("---")
st.sidebar.caption("Speed tips: keep R, n, and bootstrap B moderate; big MC uses caching + spinners.")

# -------------------------------------------------------------
# TABS scaffold
# -------------------------------------------------------------
TAB_LABELS = [
    "Start here: How to use this lab",
    "A/B tests: p-values, power & MDE",
    "CUPED: variance reduction",
    "CLT: sampling distribution of the mean",
    "LLN: running mean convergence",
    "Medians & quantiles: bootstrap variability",
    "Randomization & clustering: unit of analysis",
    "Regression vs t-test",
]
tab_readme, tab_ht, tab_cuped, tab_clt, tab_lln, tab_median, tab_cluster, tab_reg  = st.tabs(TAB_LABELS)
# -------------------------------------------------------------
# README / Start tab
# -------------------------------------------------------------
with tab_readme:
    st.subheader("Welcome — how to navigate and what to learn")
    st.markdown(
        """
**How this works**
- Use the **left sidebar** to set global defaults (replications `R`, sample size `n`, test tail, α).
- Each tab simulates many **independent experiments** and visualizes the **sampling distribution** of a statistic or test.
- Change sliders and watch **error rates, power, and variability** respond.

**Recommended path**
1. **A/B tests: p-values, power & MDE** — see Type I vs power and how `n`, Δ, and α interact.
2. **CUPED: variance reduction** — apply pre-period signals to shrink variance.
3. **Randomization & clustering** — learn why unit-of-analysis must match randomization.
4. **CLT** and **LLN** — the background: why means look Normal and converge.
5. **Medians & quantiles (bootstrap)** — why non-linear stats need bootstrap SEs.
6. **Regression vs t-test** — difference-in-means equals OLS with a treatment dummy.

**What you can do in each tab**

- **A/B tests: p-values, power & MDE**
  - *Actions:* pick two populations, set `n`, run many experiments, switch Welch/pooled, read p-value histogram, draw a power curve or solve for required `n`.
  - *Learn:* under H₀, ~α of p-values fall below α; under H₁ that fraction is **power**. How MDE, `n`, and σ trade off.

- **CUPED: variance reduction**
  - *Actions:* set Corr(X,Y), choose how θ is estimated (oracle/all/control-only), toggle centering, compare Var(Ȳ) vs Var(Ȳ*).
  - *Learn:* why centering on a fixed μ_X (or large pre-period) reduces variance, and why using in-sample X̄ doesn’t.

- **Randomization & clustering: unit of analysis**
  - *Actions:* choose ICC ρ, cluster size `m`, randomize by unit vs by cluster; compare naïve, cluster-means, and design-effect.
  - *Learn:* mismatched analysis inflates Type I error; design effect ≈ 1+(m−1)ρ cuts the effective `n`.

- **CLT: sampling distribution of the mean**
  - *Actions:* pick a population (even skewed/heavy-tailed), increase `n`, show Normal overlay/QQ plot.
  - *Learn:* means look Normal and variance shrinks like σ²/n (when Var exists).

- **LLN: running mean convergence**
  - *Actions:* plot multiple independent paths of the running mean, optionally show a 95% plug-in band.
  - *Learn:* running means stabilize around μ, with width ~1/√n (finite variance).

- **Medians & quantiles: bootstrap variability**
  - *Actions:* choose a CTR-like distribution, sample size, `q`; compare **bootstrap SE** vs MC “true” SD; (optional) CI coverage and two-sample demo.
  - *Learn:* non-linear stats don’t use σ²/n; bootstrap approximates the sampling distribution well.

- **Regression vs t-test**
  - *Actions:* draw two groups, run Welch/pooled vs OLS (with/without robust SE).
  - *Learn:* OLS with a treatment dummy **is** difference-in-means; robust SE ≈ Welch under heteroskedasticity.

**Tips**
- Keep `R` and `n` moderate for speed; bootstrap loops can be heavy.
- Welch is safer when variances differ; pooled assumes equal variances.
"""
    )

# =============================================================
# TAB 1: CLT
# =============================================================
with tab_clt:
    st.subheader("Central Limit Theorem: sampling distribution of the mean")
    st.markdown(
        """
Fix a population (random variable), draw many **samples of size** `n`, compute the **sample mean** for each sample, and visualize its **sampling distribution**. As `n` grows, the sampling distribution of the mean approaches **Normal(μ, σ²/n)** under mild conditions.
        """
    )
    left, right = st.columns([1, 1])
    with left:
        dist = st.selectbox("Population (random variable)",
                            ["Normal","Bernoulli","Poisson","Exponential","Lognormal","Pareto","Beta","Logit-Normal"],
                            key="clt_dist")
        params = {}
        if dist == "Normal":
            params["mu"] = st.number_input("μ (mean)", value=0.0)
            params["sigma"] = st.number_input("σ (sd)", value=1.0, min_value=0.0001)
        elif dist == "Bernoulli":
            params["p"] = st.slider("p (success prob)", 0.0, 1.0, 0.5, 0.01)
        elif dist == "Poisson":
            params["lam"] = st.number_input("λ (rate)", value=3.0, min_value=0.0001)
        elif dist == "Exponential":
            params["scale"] = st.number_input("scale (mean)", value=1.0, min_value=0.0001)
        elif dist == "Lognormal":
            params["meanlog"] = st.number_input("meanlog", value=0.0)
            params["sdlog"] = st.number_input("sdlog", value=1.0, min_value=0.0001)
        elif dist == "Pareto":
            params["alpha"] = st.number_input("shape α", value=3.0, min_value=0.1)
            params["scale"] = st.number_input("scale xm", value=1.0, min_value=0.0001)
        elif dist == "Beta":
            params["a"] = st.number_input("a", value=2.0, min_value=0.1)
            params["b"] = st.number_input("b", value=5.0, min_value=0.1)
        elif dist == "Logit-Normal":
            params["mu"] = st.number_input("μ (logit mean)", value=0.0)
            params["sigma"] = st.number_input("σ (logit sd)", value=1.0, min_value=0.0001)

        n = st.slider("Sample size n (per experiment)", 5, 100000, max(n_default, 100), step=5, key="clt_n")
        R = st.slider("Replications R (experiments)", 50, 50000, max(R_default, 2000), step=50, key="clt_R")
        show_normal = st.checkbox("Overlay theoretical Normal(μ, σ²/n) if defined", value=True)
        show_qq = st.checkbox("Show QQ-plot vs Normal", value=False)
        show_pop = st.checkbox("Also show population distribution (raw draws)", value=True)

    with right:
        r1, r2 = st.columns([1, 1])
        if show_pop:
            with st.spinner("Drawing population…"):
                pop_key = _params_key(params)
                pop_draws = cached_mc_draws(dist, pop_key, 40_000, seed+11)
            df_pop = pd.DataFrame({"x": pop_draws})
            pop_hist = (
                alt.Chart(df_pop)
                .transform_bin(["b0","b1"], field="x", bin=alt.Bin(maxbins=60))
                .transform_aggregate(n="count()", groupby=["b0","b1"])
                .transform_calculate(mid="(datum.b0+datum.b1)/2")
                .mark_bar()
                .encode(x=alt.X("mid:Q", title="Population values"), y="n:Q")
                .properties(height=260)
            )
            r1.altair_chart(pop_hist, use_container_width=True)

        with st.spinner("Simulating sampling distribution…"):
            means = sample_means(dist, params, n=n, R=R)
        df_means = pd.DataFrame({"sample_mean": means})
        mu_true, var_true = population_moments(dist, params)

        hist = (
            alt.Chart(df_means)
            .transform_bin(["b0","b1"], field="sample_mean", bin=alt.Bin(maxbins=60))
            .transform_aggregate(count="count()", groupby=["b0","b1"])
            .transform_calculate(mid="(datum.b0+datum.b1)/2")
            .mark_bar()
            .encode(x=alt.X("mid:Q", title="Sample mean"), y=alt.Y("count:Q", title="Count"))
            .properties(height=260)
        )
        if show_normal and (mu_true is not None) and (var_true is not None):
            xs = np.linspace(float(means.min()), float(means.max()), 200)
            se = math.sqrt(var_true/n)
            dens = stats.norm.pdf(xs, loc=mu_true, scale=se)
            k = (df_means.shape[0]*(xs.max()-xs.min())/60.0)
            curve = alt.Chart(pd.DataFrame({"x": xs, "scaled_density": dens*k})).mark_line().encode(x="x:Q", y="scaled_density:Q")
            r2.altair_chart(hist + curve, use_container_width=True)
        else:
            r2.altair_chart(hist, use_container_width=True)

        if show_qq:
            ps = np.linspace(0.005, 0.995, 101)
            qq = pd.DataFrame({"normal_quantile": stats.norm.ppf(ps),
                               "sample_quantile": np.quantile(means, ps)})
            qq_chart = alt.Chart(qq).mark_circle(size=40, opacity=0.6).encode(
                x=alt.X("normal_quantile:Q", title="Normal quantile"),
                y=alt.Y("sample_quantile:Q", title="Sample-mean quantile"),
            )
            line = alt.Chart(pd.DataFrame({"x":[-3,3],"y":[-3,3]})).mark_line().encode(x="x", y="y")
            st.altair_chart(qq_chart + line, use_container_width=True)

        emp_mu = float(np.mean(means)); emp_sd = float(np.std(means, ddof=1))
        if (mu_true is not None) and (var_true is not None):
            st.write(pd.DataFrame({
                "Quantity": ["Empirical mean","Empirical SD","Theoretical mean","Theoretical SD (σ/√n)"],
                "Value": [emp_mu, emp_sd, mu_true, math.sqrt(var_true/n)]
            }))
        else:
            st.write(pd.DataFrame({"Quantity":["Empirical mean","Empirical SD"], "Value":[emp_mu, emp_sd]}))

    explain("What to notice",
    """
- Left: raw **population** (can be skewed/heavy-tailed). Right: the **sampling distribution of the mean** concentrates and looks Normal as `n` grows.
- The sampling variance shrinks like **σ²/n**.
- If population variance is undefined (e.g., Pareto with small α), CLT can fail.
    """)

# =============================================================
# TAB 2: LLN
# =============================================================
with tab_lln:
    st.subheader("Law of Large Numbers: running mean convergence")
    st.markdown(
        r"""
We track the **running sample mean** \(\bar X_n\) as `n` increases.  
**Independent paths** = independent experiments: each path is its own i.i.d. draw sequence.
        """
    )
    left, right = st.columns([1, 1])
    with left:
        dist2 = st.selectbox("Population (random variable)",
                             ["Normal","Bernoulli","Poisson","Exponential","Lognormal","Pareto","Beta","Logit-Normal"],
                             key="lln_dist")
        params2 = {}
        if dist2 == "Normal":
            params2["mu"] = st.number_input("μ (mean)", value=0.0, key="lln_mu")
            params2["sigma"] = st.number_input("σ (sd)", value=1.0, min_value=0.0001, key="lln_sigma")
        elif dist2 == "Bernoulli":
            params2["p"] = st.slider("p (success prob)", 0.0, 1.0, 0.5, 0.01, key="lln_p")
        elif dist2 == "Poisson":
            params2["lam"] = st.number_input("λ (rate)", value=3.0, min_value=0.0001, key="lln_lam")
        elif dist2 == "Exponential":
            params2["scale"] = st.number_input("scale (mean)", value=1.0, min_value=0.0001, key="lln_scale")
        elif dist2 == "Lognormal":
            params2["meanlog"] = st.number_input("meanlog", value=0.0, key="lln_meanlog")
            params2["sdlog"] = st.number_input("sdlog", value=1.0, min_value=0.0001, key="lln_sdlog")
        elif dist2 == "Pareto":
            params2["alpha"] = st.number_input("shape α", value=3.0, min_value=0.1, key="lln_alpha")
            params2["scale"] = st.number_input("scale xm", value=1.0, min_value=0.0001, key="lln_xm")
        elif dist2 == "Beta":
            params2["a"] = st.number_input("a", value=2.0, min_value=0.1, key="lln_a")
            params2["b"] = st.number_input("b", value=5.0, min_value=0.1, key="lln_b")
        elif dist2 == "Logit-Normal":
            params2["mu"] = st.number_input("μ (logit mean)", value=0.0, key="lln_lmu")
            params2["sigma"] = st.number_input("σ (logit sd)", value=1.0, min_value=0.0001, key="lln_lsig")

        n_path = st.slider("Max draws per path", 50, 200000, 5000, step=50, key="lln_npath")
        num_paths = st.slider("Number of independent paths", 1, 50, 10, step=1, key="lln_paths")
        show_ci = st.checkbox("Show 95% CI band (plugin)", value=False, key="lln_show_ci")

    with right:
        with st.spinner("Simulating paths…"):
            paths = draw_from_population(dist2, params2, size=(num_paths, n_path))
            running_means = np.cumsum(paths, axis=1)/np.arange(1, n_path+1)
        # Decimate plotted points for speed
        max_points = 2500; step = max(1, n_path//max_points)
        idx = np.arange(1, n_path+1, step)
        rm_plot = running_means[:, idx-1]
        df_lines = pd.DataFrame(rm_plot).reset_index().melt(id_vars="index", var_name="t", value_name="running_mean")
        df_lines.rename(columns={"index":"path"}, inplace=True)
        df_lines["t"] = idx[df_lines["t"].astype(int)]
        mu_true2, var_true2 = population_moments(dist2, params2)
        base = alt.Chart(df_lines).mark_line(opacity=0.55).encode(
            x=alt.X("t:Q", title="n (draws so far)"),
            y=alt.Y("running_mean:Q", title="Running sample mean"),
            detail="path:N",
        )
        charts = [base]
        if mu_true2 is not None:
            charts.append(alt.Chart(pd.DataFrame({"mu":[mu_true2]})).mark_rule().encode(y="mu:Q"))
        if show_ci and (var_true2 is not None):
            se_t = np.sqrt(var_true2)/np.sqrt(np.maximum(1, idx))
            band = pd.DataFrame({"t": idx,
                                 "lo": (mu_true2 if mu_true2 is not None else rm_plot.mean(axis=0)) - 1.96*se_t,
                                 "hi": (mu_true2 if mu_true2 is not None else rm_plot.mean(axis=0)) + 1.96*se_t})
            charts.append(alt.Chart(band).mark_area(opacity=0.15).encode(x="t:Q", y="lo:Q", y2="hi:Q"))
        st.altair_chart(alt.layer(*charts).properties(height=350), use_container_width=True)

    explain("What to notice",
    """
- Paths wobble but settle near μ as `n` grows (LLN = convergence).
- The optional CI shrinks ∝ 1/√n (finite variance case).
- We plot every k-th point for speed; this does not change the statistical story.
    """)

# =============================================================
# TAB 3: Hypothesis testing + Power / MDE
# =============================================================
with tab_ht:
    st.subheader("A/B tests: p-values, power & MDE")
    st.markdown(
        """
We repeat **independent experiments**: draw a control and a treatment sample, run a two-sample t-test each time, and study the distribution of p-values. Under **H₀ (Δ=0)**, about **α** of p-values fall below α (Type I error). Under **H₁ (Δ≠0)**, that fraction is the **power**.
        """
    )

    a, b = st.columns([1, 1])
    with a:
        st.markdown("**Control population**")
        distA = st.selectbox("Distribution A (control)", ["Normal","Bernoulli","Poisson","Exponential","Lognormal","Beta"], key="ht_distA")
        paramsA = {}
        if distA == "Normal":
            paramsA["mu"] = st.number_input("A: μ", value=0.0, key="A_mu")
            paramsA["sigma"] = st.number_input("A: σ", value=1.0, min_value=0.0001, key="A_sigma")
        elif distA == "Bernoulli":
            paramsA["p"] = st.slider("A: p", 0.0, 1.0, 0.5, 0.01, key="A_p")
        elif distA == "Poisson":
            paramsA["lam"] = st.number_input("A: λ", value=3.0, min_value=0.0001, key="A_lam")
        elif distA == "Exponential":
            paramsA["scale"] = st.number_input("A: scale (mean)", value=1.0, min_value=0.0001, key="A_scale")
        elif distA == "Lognormal":
            paramsA["meanlog"] = st.number_input("A: meanlog", value=0.0, key="A_meanlog")
            paramsA["sdlog"] = st.number_input("A: sdlog", value=1.0, min_value=0.0001, key="A_sdlog")
        elif distA == "Beta":
            paramsA["a"] = st.number_input("A: a", value=2.0, min_value=0.1, key="A_a")
            paramsA["b"] = st.number_input("A: b", value=5.0, min_value=0.1, key="A_b")

        st.markdown("**Treatment population**")
        distB = st.selectbox("Distribution B (treatment)", ["Normal","Bernoulli","Poisson","Exponential","Lognormal","Beta"], key="ht_distB")
        force_equal_means = st.checkbox("Force treatment mean equal to control mean (H₀: Δ=0)", value=False, key="ht_equal")
        paramsB = {}
        if distB == "Normal":
            paramsB["mu"] = st.number_input("B: μ", value=(paramsA.get("mu",0.0) if force_equal_means else 0.5), key="B_mu")
            paramsB["sigma"] = st.number_input("B: σ", value=paramsA.get("sigma",1.0), min_value=0.0001, key="B_sigma")
        elif distB == "Bernoulli":
            paramsB["p"] = st.slider("B: p", 0.0, 1.0, (paramsA.get("p",0.5) if force_equal_means else 0.6), 0.01, key="B_p")
        elif distB == "Poisson":
            paramsB["lam"] = st.number_input("B: λ", value=(paramsA.get("lam",3.0) if force_equal_means else 3.5), min_value=0.0001, key="B_lam")
        elif distB == "Exponential":
            paramsB["scale"] = st.number_input("B: scale (mean)", value=(paramsA.get("scale",1.0) if force_equal_means else 1.2), min_value=0.0001, key="B_scale")
        elif distB == "Lognormal":
            paramsB["meanlog"] = st.number_input("B: meanlog", value=(paramsA.get("meanlog",0.0) if force_equal_means else 0.3), key="B_meanlog")
            paramsB["sdlog"] = st.number_input("B: sdlog", value=paramsA.get("sdlog",1.0), min_value=0.0001, key="B_sdlog")
        elif distB == "Beta":
            paramsB["a"] = st.number_input("B: a", value=(paramsA.get("a",2.0) if force_equal_means else 3.0), min_value=0.1, key="B_a")
            paramsB["b"] = st.number_input("B: b", value=paramsA.get("b",5.0), min_value=0.1, key="B_b")

        n_ht = st.slider("n per group", 5, 100000, n_default, step=5, key="ht_n")
        R_ht = st.slider("Replications R", 50, 50000, R_default, step=50, key="ht_R")
        alpha_ht = st.select_slider("α (significance)", options=[0.01,0.02,0.025,0.05,0.10], value=alpha_default, key="ht_alpha")
        ttype = st.radio("t-test type", ("Welch (unequal var)","Pooled (equal var)"), index=0, key="ht_ttype")

        st.markdown("**Population overlap options**")
        popN = st.slider("MC draws per group (overlap plot)", 10_000, 300_000, 80_000, 10_000, key="ht_popN")
        lock_bw = st.checkbox("Use common KDE bandwidth (visual parity under H₀)", True, key="ht_lockbw")
        show_pdf = st.checkbox("Overlay theoretical PDFs (if available)", True, key="ht_pdf")

    with b:
        # ---------- Population overlap ----------
        with st.spinner("Drawing population overlap…"):
            popA = cached_mc_draws(distA, _params_key(paramsA), popN, seed+21)
            popB = cached_mc_draws(distB, _params_key(paramsB), popN, seed+22)
        df_pop = pd.DataFrame({"x": np.concatenate([popA,popB]),
                               "group": ["A (control)"]*popN + ["B (treat)"]*popN})
        if lock_bw:
            pooled_sd = float(np.sqrt(0.5*(np.var(popA, ddof=1) + np.var(popB, ddof=1))))
            bw = 1.06 * pooled_sd * (popN ** (-1/5))
            dens = alt.Chart(df_pop).transform_density("x", groupby=["group"], as_=["x","density"], bandwidth=bw)\
                                    .mark_area(opacity=0.35).encode(x=alt.X("x:Q", title="Population values"),
                                                                     y=alt.Y("density:Q", stack=None),
                                                                     color="group:N")
        else:
            dens = alt.Chart(df_pop).transform_density("x", groupby=["group"], as_=["x","density"])\
                                    .mark_area(opacity=0.35).encode(x=alt.X("x:Q", title="Population values"),
                                                                     y=alt.Y("density:Q", stack=None),
                                                                     color="group:N")
        charts = [dens]

        def _pdf_for(dist, params, xs):
            d = dist.lower()
            if d == "normal":   return stats.norm.pdf(xs, loc=params["mu"], scale=params["sigma"])
            if d == "exponential": return stats.expon.pdf(xs, scale=params["scale"])
            if d == "lognormal":   return stats.lognorm(s=params["sdlog"], scale=np.exp(params["meanlog"])).pdf(xs)
            if d == "beta":        return stats.beta(params["a"], params["b"]).pdf(xs)
            return None

        if show_pdf:
            x_lo = float(df_pop["x"].quantile(0.001)); x_hi = float(df_pop["x"].quantile(0.999))
            xs = np.linspace(x_lo, x_hi, 400)
            for lab, dist, par in [("A (control)", distA, paramsA), ("B (treat)", distB, paramsB)]:
                pdf = _pdf_for(dist, par, xs)
                if pdf is not None and np.all(np.isfinite(pdf)):
                    charts.append(alt.Chart(pd.DataFrame({"x":xs,"density":pdf,"group":lab}))
                                  .mark_line(strokeDash=[4,2]).encode(x="x:Q", y="density:Q", color="group:N"))
        st.altair_chart(alt.layer(*charts).properties(height=200), use_container_width=True)

        # ---------- R independent experiments (vectorized) ----------
        with st.spinner("Running repeated experiments…"):
            X = draw_from_population(distA, paramsA, size=(R_ht, n_ht))
            Y = draw_from_population(distB, paramsB, size=(R_ht, n_ht))
            muA, _ = population_moments(distA, paramsA); muB, _ = population_moments(distB, paramsB)
            if muA is None: muA = float(np.mean(popA))
            if muB is None: muB = float(np.mean(popB))
            diff_true = float(muB - muA)
            if ttype.startswith("Welch"):
                _, pvals = two_sample_welch_ttest_matrix(X, Y)
            else:
                _, pvals = pooled_ttest_matrix(X, Y)
            est_diffs = Y.mean(axis=1) - X.mean(axis=1)

        reject = pvals < alpha_ht
        is_null = abs(diff_true) < 1e-12
        frac_sig = float(np.mean(reject))
        st.markdown("**Across R experiments (replications)**")
        st.write(pd.DataFrame({
            "True difference Δ": [diff_true],
            ("Empirical Type I (≈α)" if is_null else "Empirical power (1−β)"): [frac_sig],
            "Mean estimated diff": [float(np.mean(est_diffs))],
            "SD of estimated diff": [float(np.std(est_diffs, ddof=1))],
        }))

        st.markdown("**p-value distribution across replications**")
        p_hist = alt.Chart(pd.DataFrame({"p": pvals}))\
                 .transform_bin(["b0","b1"], field="p", bin=alt.Bin(maxbins=40, extent=[0,1]))\
                 .transform_aggregate(n="count()", groupby=["b0","b1"])\
                 .transform_calculate(mid="(datum.b0+datum.b1)/2")\
                 .mark_bar().encode(x=alt.X("mid:Q", title="p-value", scale=alt.Scale(domain=[0,1])),
                                    y=alt.Y("n:Q", title="count"))
        st.altair_chart(p_hist.properties(height=180), use_container_width=True)

        # Three concrete single experiments
        st.markdown("**Three example experiments (observed samples & their p-values)**")
        if R_ht >= 3:
            idxs = np.random.choice(R_ht, size=3, replace=False)
            charts = []
            for j, i in enumerate(idxs):
                df_ex = pd.DataFrame({"x": np.concatenate([X[i,:], Y[i,:]]), "group": ["A"]*n_ht + ["B"]*n_ht})
                dens_ex = alt.Chart(df_ex).transform_density("x", groupby=["group"], as_=["x","density"])\
                                          .mark_area(opacity=0.35).encode(
                        x=alt.X("x:Q", title=f"Experiment {j+1}"), y=alt.Y("density:Q", stack=None), color="group:N"
                ).properties(width=220, height=160)
                charts.append(dens_ex)
            st.altair_chart(alt.hconcat(*charts), use_container_width=True)

    st.markdown("---")
    st.markdown("**Power & MDE (Normal approximation)** — uses a common SD (σ) estimate.")
    c1,c2,c3 = st.columns(3)
    with c1:
        sigma_est = estimate_sigma_from_dist(distA, paramsA)
        st.write(f"Estimated SD (σ) ≈ {sigma_est:.4f}")
    with c2:
        mde = st.number_input("MDE (Δ) for power curve", value=0.2, key="ht_mde")
        n_for_curve = st.slider("n per group for power point", 5, 100000, n_ht, step=5, key="ht_ncurve")
        st.write(f"Approx power at n={n_for_curve}: {normal_power_two_sample(mde, sigma_est, n_for_curve, alpha_ht, TWO_SIDED):.3f}")
    with c3:
        target_power = st.slider("Target power", 0.50, 0.99, 0.80, 0.01, key="ht_target_power")
        st.write(f"n per group to hit {target_power:.0%} power: {solve_n_for_power(max(mde,1e-9), sigma_est, alpha_ht, target_power, TWO_SIDED)}")
    ns = np.linspace(max(5, n_for_curve//5), n_for_curve*2, 30, dtype=int)
    df_pow = pd.DataFrame({"n": ns, "power": [normal_power_two_sample(mde, sigma_est, int(n), alpha_ht, TWO_SIDED) for n in ns]})
    st.altair_chart(alt.Chart(df_pow).mark_line(point=True).encode(x="n:Q", y=alt.Y("power:Q", scale=alt.Scale(domain=[0,1]))).properties(height=240),
                    use_container_width=True)

    explain("Notes on tests & power",
    """
- **Welch** (unequal variances) is robust; **pooled** assumes equal variances.
- Under **H₀ (Δ=0)**, p-values are **Uniform(0,1)** → about **α** of runs reject.
- Under **H₁ (Δ≠0)**, the rejection fraction is **power** (1−β).
- **MDE** asks: “smallest Δ detectable at given `n` and α with high power?”
    """)

# =============================================================
# TAB 4: Empirical variability (quantiles via bootstrap)
# =============================================================
with tab_median:
    st.subheader("Medians & quantiles: bootstrap variability")
    st.markdown(
        """
For **non-linear** statistics like the **median** or a general **q-quantile**, the mean-style variance formula \(s^2/n\) is **wrong**.  
We compare three variance estimates for the chosen quantile:
1) **Bootstrap** (empirical), 2) **Naïve mean-style** \(s^2/n\) (for teaching: this is inappropriate), and 3) the **true sampling variance** via Monte-Carlo from the population.
        """
    )

    a, b = st.columns([1, 1])
    with a:
        ctr_dist = st.selectbox("Population", ["Beta","Logit-Normal","Mixture: Zero-inflated Beta"], key="med_dist")
        ctr_params = {}; zero_pi = 0.0
        if ctr_dist == "Beta":
            ctr_params["a"] = st.number_input("a", value=2.0, min_value=0.1, key="ctr_a")
            ctr_params["b"] = st.number_input("b", value=8.0, min_value=0.1, key="ctr_b")
        elif ctr_dist == "Logit-Normal":
            ctr_params["mu"] = st.number_input("logit μ", value=-1.5, key="ctr_lmu")
            ctr_params["sigma"] = st.number_input("logit σ", value=1.0, min_value=0.0001, key="ctr_lsig")
        else:
            ctr_params["a"] = st.number_input("Beta a", value=1.2, min_value=0.1, key="ctr_mixa")
            ctr_params["b"] = st.number_input("Beta b", value=20.0, min_value=0.1, key="ctr_mixb")
            zero_pi = st.slider("Zero-inflation π (mass at 0)", 0.0, 0.9, 0.2, 0.05)

        n_med = st.slider("Sample size n", 10, 200000, 5000, step=10, key="med_n")
        B = st.slider("Bootstrap resamples B", 100, 5000, 1000, step=100, key="med_B")
        q = st.slider("Quantile q", 0.10, 0.90, 0.50, 0.05, key="med_q")
        R_truth = st.slider("Monte-Carlo replications for true sampling variance", 100, 5000, 1000, 100, key="med_truthR")
        R_cov = st.slider("Replications for CI coverage demo (set 0 to skip)", 0, 2000, 200, 50, key="med_Rcov")
        method = st.radio("Bootstrap CI method", ("Percentile","Basic"), index=0, key="med_ci")

    with b:
        def draw_ctr(size):
            if ctr_dist == "Beta":
                return draw_from_population("beta", ctr_params, size)
            elif ctr_dist == "Logit-Normal":
                return draw_from_population("logit-normal", ctr_params, size)
            # mixture
            rng = np.random.default_rng()
            z = rng.random(size) < zero_pi
            beta_part = rng.beta(ctr_params["a"], ctr_params["b"], size)
            beta_part[z] = 0.0
            return beta_part

        def bootstrap_ci_quantile(sample: np.ndarray, q=0.5, B=1000, method="Percentile"):
            n = len(sample)
            idx = np.random.randint(0, n, size=(B, n))
            boots = np.quantile(sample[idx], q, axis=1) if q!=0.5 else np.median(sample[idx], axis=1)
            est = float(np.quantile(sample, q)) if q!=0.5 else float(np.median(sample))
            se = float(np.std(boots, ddof=1))
            lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
            if method.lower().startswith("basic"): lo, hi = 2*est - hi, 2*est - lo
            return est, se, lo, hi, boots

        with st.spinner("Single experiment + bootstrap…"):
            sample = draw_ctr(n_med)
            est, se_boot, lo, hi, boots = bootstrap_ci_quantile(sample, q=q, B=B, method=method)

        # "True" population quantile (analytic when possible; else MC)
        if ctr_dist == "Beta":
            true_q = float(stats.beta(ctr_params["a"], ctr_params["b"]).ppf(q))
        elif ctr_dist == "Logit-Normal":
            true_q = cached_true_quantile("logit-normal", _params_key(ctr_params), q, seed+31)
        else:
            with st.spinner("Estimating population quantile by Monte Carlo…"):
                true_q = float(np.quantile(draw_ctr(200_000), q))

        # True sampling variance of the estimator at size n (MC)
        with st.spinner("Estimating true sampling variance…"):
            ests_truth = [float(np.quantile(draw_ctr(n_med), q)) for _ in range(R_truth)]
        true_sd_stat = float(np.std(ests_truth, ddof=1))

        # Naïve mean-style variance (for teaching: wrong for quantiles)
        naive_var_meanstyle = float(np.var(sample, ddof=1) / n_med)

        # Display & distributions
        st.markdown("**Single experiment summary**")
        st.write(pd.DataFrame({
            "Statistic": ["Quantile estimate", "Bootstrap SE", "True SD of estimator (MC)", "95% CI lower", "95% CI upper", "True population quantile"],
            "Value": [est, se_boot, true_sd_stat, lo, hi, true_q]
        }))

        st.markdown("**Variance estimates for this statistic**")
        df_var = pd.DataFrame({
            "Estimator": ["Bootstrap (SE²)","Naïve mean-style (s²/n)","True sampling (MC)"],
            "Variance": [se_boot**2, naive_var_meanstyle, true_sd_stat**2]
        })
        st.altair_chart(
            alt.Chart(df_var).mark_bar().encode(x="Estimator:N", y="Variance:Q"),
            use_container_width=True
        )

        # Sampling distributions (bootstrap vs MC)
        df_boot = pd.DataFrame({"value": boots, "which": ["Bootstrap"]*len(boots)})
        df_truth = pd.DataFrame({"value": ests_truth, "which": ["Monte Carlo (true)"]*len(ests_truth)})
        st.altair_chart(
            alt.Chart(pd.concat([df_boot, df_truth]))
              .transform_density("value", groupby=["which"], as_=["value","density"])
              .mark_area(opacity=0.35)
              .encode(x=alt.X("value:Q", title=f"Sampling distribution of q={q:.2f}"),
                      y=alt.Y("density:Q", stack=None), color="which:N")
              .properties(height=220),
            use_container_width=True
        )

        # Optional CI coverage simulation
        if R_cov > 0:
            with st.spinner("CI coverage simulation…"):
                covers = []
                for _ in range(R_cov):
                    s = draw_ctr(n_med)
                    e, _, L, H, _ = bootstrap_ci_quantile(s, q=q, B=max(100, B//2), method=method)
                    covers.append(1.0 if (L <= true_q <= H) else 0.0)
            st.write(pd.DataFrame({"Coverage (fraction of CIs containing true value)": [float(np.mean(covers))]}))

    explain("Why bootstrap for quantiles?",
    """
- The variance of a **quantile estimator** depends on the **density at the true quantile**; generic \(s^2/n\) does **not** apply.
- Bootstrap resampling approximates the **sampling distribution** from the observed sample without fragile analytic formulas.
- We explicitly compare **bootstrap variance**, the incorrect **mean-style** \(s^2/n\), and the **true** variance from Monte-Carlo to calibrate intuition.
    """)

# =============================================================
# TAB 5: Randomization & unit-of-analysis (clustering)
# =============================================================
with tab_cluster:
    st.subheader("Randomization level vs unit-of-analysis")
    st.markdown(
        """
We simulate **clustered data** with intra-class correlation (ICC). Compare:
- **Naïve individual** analysis (ignores clustering),
- **Cluster-means** analysis (analyze one mean per cluster),
- **Design-effect** adjustment $\mathrm{DE} \approx 1 + (m-1)\rho$ applied to the naïve SE.

**Concrete search example:** users issue many searches → **users are clusters**.  
- Randomize **by user** and analyze **CTR/user** → correct.  
- Randomize **by search** but analyze **CTR/user as if users were in a single arm** → wrong: exposures mix, SEs are biased. Either analyze at the **search** level with cluster-robust SEs or aggregate consistent with assignment.
        """
    )
    a, b = st.columns([1, 1])
    with a:
        K = st.slider("Number of clusters K", 4, 200, 50, step=2, key="cl_K")
        m = st.slider("Units per cluster m", 2, 1000, 50, step=2, key="cl_m")
        rho = st.slider("ICC ρ", 0.0, 0.3, 0.05, 0.01, key="cl_rho")
        delta = st.number_input("True treatment effect Δ (added to treated)", value=0.0, step=0.05, key="cl_delta")
        R_c = st.slider("Replications R", 100, 10000, 1000, step=100, key="cl_R")
        alpha_c = st.select_slider("α", options=[0.01,0.02,0.025,0.05,0.10], value=0.05, key="cl_alpha")
        scheme = st.radio("Randomization scheme", ("Unit-level","Cluster-level"), index=1, key="cl_scheme")

    with b:
        def simulate_once():
            sigma_c2 = rho; sigma_w2 = max(1e-9, 1-rho)
            c_eff = np.random.normal(0, np.sqrt(sigma_c2), K)
            if scheme.startswith("Cluster"):
                treat_clusters = np.random.binomial(1, 0.5, K)
                T = np.repeat(treat_clusters, m)
            else:
                T = np.random.binomial(1, 0.5, K*m)
            eps = np.random.normal(0, np.sqrt(sigma_w2), K*m)
            g = np.repeat(np.arange(K), m)
            y0 = c_eff[g] + eps
            y = y0 + delta*T
            return y, T, g

        def analyze(y, T, g):
            y0 = y[T==0]; y1 = y[T==1]
            _, p_naive = stats.ttest_ind(y0, y1, equal_var=False)
            # cluster-means
            df = pd.DataFrame({"y": y, "T": T, "g": g})
            cl_means = df.groupby(["g","T"], as_index=False).agg(y=("y","mean"))
            y0c = cl_means.loc[cl_means["T"]==0, "y"].values
            y1c = cl_means.loc[cl_means["T"]==1, "y"].values
            p_cluster = np.nan
            if len(y0c)>1 and len(y1c)>1:
                _, p_cluster = stats.ttest_ind(y0c, y1c, equal_var=False)
            # design-effect adjusted z using naive diff
            diff_hat = float(np.mean(y[T==1]) - np.mean(y[T==0]))
            n0, n1 = np.sum(T==0), np.sum(T==1)
            s0, s1 = np.var(y[T==0], ddof=1), np.var(y[T==1], ddof=1)
            se_naive = math.sqrt(s0/n0 + s1/n1)
            de = 1 + (m-1)*rho
            se_adj = se_naive * math.sqrt(de)
            z = abs(diff_hat)/max(se_adj,1e-12)
            p_adj = 2*(1 - stats.norm.cdf(z))
            return p_naive, p_cluster, p_adj

        with st.spinner("Simulating clustered experiments…"):
            p1, p2, p3 = [], [], []
            for _ in range(R_c):
                y, T, g = simulate_once()
                a1, a2, a3 = analyze(y, T, g); p1.append(a1); p2.append(a2); p3.append(a3)

        def rate(ps): return float(np.mean(np.array(ps) < alpha_c))
        label = "Type I error" if abs(delta)<1e-12 else "Power"
        df_rates = pd.DataFrame({"Method":["Naïve individual","Cluster means","Design-effect adj"],
                                 label:[rate(p1), rate(p2), rate(p3)]})
        st.altair_chart(alt.Chart(df_rates).mark_bar().encode(x="Method:N", y=f"{label}:Q").properties(height=250),
                        use_container_width=True)

        # Display effective n (design effect)
        n_total = K*m
        n_eff = n_total / (1 + (m-1)*rho)
        st.caption(f"Design effect DE ≈ 1 + (m−1)ρ ⇒ effective sample size ≈ {n_eff:.1f} (out of {n_total}).")

    explain("Takeaways",
    """
- **Unit of randomization** must match (or be accounted for in) the **unit of analysis**.
- **Ignoring clustering** understates variance and inflates **Type I error**.
- Two robust choices: analyze **cluster means**, or adjust the naïve SE by a **design effect** \(1+(m−1)ρ\) (or use cluster-robust SEs).
    """)

# =============================================================
# TAB 6: Regression vs t-test
# =============================================================
with tab_reg:
    st.subheader("Regression and two-sample t-test: the same object")
    st.markdown("OLS with an intercept and a treatment dummy equals **difference in means**; with robust SEs it aligns with Welch’s test.")
    a, b = st.columns([1,1])
    with a:
        distR = st.selectbox("Population for both groups", ["Normal","Lognormal"], key="reg_dist")
        paramsR = {}
        if distR == "Normal":
            paramsR["mu"] = st.number_input("μ base", value=0.0, key="reg_mu")
            paramsR["sigma"] = st.number_input("σ base", value=1.0, min_value=0.0001, key="reg_sigma")
        else:
            paramsR["meanlog"] = st.number_input("meanlog base", value=0.0, key="reg_meanlog")
            paramsR["sdlog"]  = st.number_input("sdlog base", value=0.75, min_value=0.0001, key="reg_sdlog")
        deltaR = st.number_input("Additive shift for treatment Δ", value=0.5, key="reg_delta")
        nR = st.slider("n per group", 5, 100000, 200, step=5, key="reg_n")
        use_robust = st.checkbox("Use robust (HC1) SE in regression", value=True, key="reg_robust")

    with b:
        x0 = draw_from_population(distR, paramsR, nR)
        prm = paramsR.copy()
        if distR == "Normal": prm["mu"] = prm.get("mu",0.0) + deltaR; x1 = draw_from_population(distR, prm, nR)
        else:                  x1 = draw_from_population(distR, paramsR, nR) + deltaR

        t_stat_w, p_w = two_sample_welch_ttest(x0, x1)
        t_stat_p, p_p = pooled_ttest(x0, x1)

        y = np.concatenate([x0, x1])
        T = np.array([0]*nR + [1]*nR)
        X = np.column_stack([np.ones_like(T), T])
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ (X.T @ y)
        resid = y - X @ beta; n = len(y); k = X.shape[1]
        if use_robust:
            S = np.diag(resid**2); V = XtX_inv @ (X.T @ S @ X) @ XtX_inv * (n/(n-k))
        else:
            sigma2 = (resid @ resid)/(n-k); V = XtX_inv * sigma2
        se = np.sqrt(np.diag(V)); t_beta1 = beta[1]/se[1]; p_beta1 = 2*(1 - stats.t.cdf(abs(t_beta1), df=n-k))

        st.write(pd.DataFrame({
            "Item": ["Group mean (control)","Group mean (treat)","Diff-in-means","Welch t","Welch p","Pooled t","Pooled p","OLS coef (treat)","OLS SE","OLS t","OLS p"],
            "Value": [float(np.mean(x0)), float(np.mean(x1)), float(np.mean(x1)-np.mean(x0)),
                      t_stat_w, p_w, t_stat_p, p_p,
                      float(beta[1]), float(se[1]), float(t_beta1), float(p_beta1)]
        }))

        df_sc = pd.DataFrame({"y": y, "T": T})
        df_sc["Tx"] = df_sc["T"] + (np.random.rand(len(df_sc)) - 0.5)*0.1
        means_df = pd.DataFrame({"T":[0,1], "y":[float(np.mean(x0)), float(np.mean(x1))]})
        scatter = alt.Chart(df_sc).mark_circle(opacity=0.4).encode(x=alt.X("Tx:Q", title="Treatment indicator (jittered)"), y="y:Q")
        line = alt.Chart(means_df).mark_line(point=True).encode(x=alt.X("T:Q", scale=alt.Scale(domain=[-0.1,1.1])), y="y:Q")
        st.altair_chart((scatter+line).properties(height=240), use_container_width=True)

# =============================================================
# TAB 7: CUPED variance reduction
# =============================================================
with tab_cuped:
    st.subheader("CUPED: variance reduction via a pre-period covariate")
    st.markdown(
        r"""
CUPED uses a covariate \(X\) measured **before** the experiment to reduce the variance of the estimator.  
We transform $Y$ to $Y^* = Y - \theta \big(X - \mu_X\big)$.

- If $\mu_X$ is the **population** mean (e.g., known or well-estimated from a large **pre-period**), the sample mean $\bar Y^*$ has **lower variance** than $\bar Y$.
- If you center at the **in-sample** mean $\bar X$, then $\bar Y^*=\bar Y$ exactly (no variance change) — that's why you previously saw no reduction.

For standardized bivariate normal with $\operatorname{Var}(X)=\operatorname{Var}(Y)=1$ and $\operatorname{Cov}(X,Y)=\rho$, the oracle choice is $\theta=\rho$ and
$\operatorname{Var}(\bar Y^*) \approx (1-\rho^2)\operatorname{Var}(\bar Y)$.
        """
    )
    a, b = st.columns([1, 1])
    with a:
        rho_c = st.slider("Corr(X, Y)", 0.0, 0.95, 0.6, 0.05, key="cu_rho")
        n_c = st.slider("Sample size n", 10, 200000, 1000, step=10, key="cu_n")
        R_cup = st.slider("Replications R", 100, 20000, 4000, step=100, key="cu_R")
        theta_src = st.radio("θ computed from", ("Pooled (this sample)", "Control-only half (this sample)", "Oracle θ=ρ"), index=0, key="cu_theta_src")
        muX_src = st.radio("Center X at μ_X", ("Pre-period estimate", "True μ_X=0 (oracle)", "In-sample X̄ (no reduction)"), index=0, key="cu_mu")
        show_scatter = st.checkbox("Show 3 example (X,Y) scatters with sample correlations", value=True, key="cu_show_scatter")

    with b:
        with st.spinner("Simulating CUPED replications…"):
            rng = np.random.default_rng(seed+901)
            Z1 = rng.normal(0, 1, size=(R_cup, n_c))
            Z2 = rng.normal(0, 1, size=(R_cup, n_c))
            X = Z1
            Y0 = rho_c*Z1 + np.sqrt(max(1e-12, 1 - rho_c**2))*Z2
            Y = Y0  # mean 0; add constant if you want; it cancels in diffs

            # θ
            if theta_src.startswith("Oracle"):
                theta = np.full(R_cup, rho_c)
            else:
                if theta_src.startswith("Control"):
                    mask = (np.arange(n_c) % 2)==0
                    Xt, Yt = X[:,mask], Y[:,mask]
                else:
                    Xt, Yt = X, Y
                Xc = Xt - Xt.mean(axis=1, keepdims=True)
                Yc = Yt - Yt.mean(axis=1, keepdims=True)
                theta = (Xc*Yc).mean(axis=1) / np.maximum((Xc**2).mean(axis=1), 1e-12)

            # μ_X reference
            if muX_src.startswith("True"):
                muX = 0.0
            elif muX_src.startswith("Pre-period"):
                # simulate a large pre-period to estimate μ_X (close to 0)
                muX = float(np.mean(rng.normal(0,1,size=200_000)))
            else:
                muX = None  # use in-sample X̄ (no reduction)

            # CUPED transform
            if muX is None:
                X_center = X - X.mean(axis=1, keepdims=True)  # in-sample -> no change to mean variance
            else:
                X_center = X - muX
            Y_star = Y - theta[:, None] * X_center

            # Compare sampling distributions of means
            Ybar = Y.mean(axis=1)
            Ystar_bar = Y_star.mean(axis=1)
            df_c = pd.DataFrame({"mean": np.concatenate([Ybar, Ystar_bar]),
                                 "which": ["Y"]*len(Ybar) + ["Y* (CUPED)"]*len(Ystar_bar)})
            st.altair_chart(
                alt.Chart(df_c).transform_density("mean", groupby=["which"], as_=["mean","density"]).mark_area(opacity=0.35)
                .encode(x=alt.X("mean:Q", title="Sample mean across replications"), y=alt.Y("density:Q", stack=None), color="which:N")
                .properties(height=260),
                use_container_width=True
            )

            emp_var_Y  = float(np.var(Ybar, ddof=1))
            emp_var_Ys = float(np.var(Ystar_bar, ddof=1))
            expected_reduction = 100.0 * (rho_c**2)  # oracle θ, μ_X=0 case

            st.write(pd.DataFrame({
                "Quantity": ["E[Ȳ] (empirical)","Var(Ȳ) (empirical)","E[Ȳ*] (empirical)","Var(Ȳ*) (empirical)","Observed variance reduction %","Expected reduction % (oracle θ, μ_X=0)"],
                "Value": [float(np.mean(Ybar)), emp_var_Y, float(np.mean(Ystar_bar)), emp_var_Ys,
                          100.0*(1 - emp_var_Ys / max(emp_var_Y,1e-12)), expected_reduction]
            }))

            if show_scatter:
                ids = np.random.choice(R_cup, size=3, replace=False)
                charts = []
                for j,i in enumerate(ids):
                    df_xy = pd.DataFrame({"X": X[i,:], "Y": Y[i,:]})
                    r = float(np.corrcoef(X[i,:], Y[i,:])[0,1])
                    charts.append(alt.Chart(df_xy).mark_circle(opacity=0.4).encode(x="X:Q", y="Y:Q")
                                  .properties(width=220, height=180, title=f"Rep {j+1}: r≈{r:.2f}"))
                st.altair_chart(alt.hconcat(*charts), use_container_width=True)

    explain("Takeaways",
    """
- Centering $X$ at a **fixed** $\mu_X$ (true or pre-period) yields real variance reduction; centering at in-sample $\bar X$ yields none.
- With oracle $\theta=\rho$ and standardized variables, the variance factor is about $1-\rho^2$.
- In A/B tests, apply CUPED **separately by arm** and compare transformed means (or include $X$ in a regression with pre-period mean subtracted).
    """)

# -------------------------------------------------------------
# Footer
# -------------------------------------------------------------
st.markdown(
    """
---
**Run locally**
1. `pip install streamlit numpy scipy pandas altair`
2. `streamlit run app.py`

**requirements.txt**
streamlit>=1.37
numpy>=1.24
scipy>=1.10
pandas>=1.5
altair>=5.0
"""
)
