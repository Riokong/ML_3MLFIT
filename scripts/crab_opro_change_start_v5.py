# v5: OPRO warm-start + Minuit HESSE for parameter uncertainties

from hawc_hal import HAL, HealpixConeROI
import matplotlib.pyplot as plt
from threeML import *
from openai import OpenAI
import json
import re
import os
import random

# ── Ollama client (local) ─────────────────────────────────────────────────────
client = OpenAI(base_url="http://localhost:11435/v1", api_key="ollama")

# ── HAWC setup ────────────────────────────────────────────────────────────────
ra_crab, dec_crab = 83.63, 22.02

roi = HealpixConeROI(data_radius=3.0,
                     model_radius=8.0,
                     ra=ra_crab,
                     dec=dec_crab)

maptree  = "HAWC_9bin_507days_crab_data.hd5"
response = "HAWC_9bin_507days_crab_response.hd5"

hawc = HAL("HAWC", maptree, response, roi, flat_sky_pixels_size=0.1)
hawc.set_active_measurements(1, 9)

# ── Model ─────────────────────────────────────────────────────────────────────
spectrum = Log_parabola()
source   = PointSource("crab", ra=ra_crab, dec=dec_crab, spectral_shape=spectrum)

spectrum.piv       = 7 * u.TeV
spectrum.piv.fix   = True
spectrum.K         = 1e-23 / (u.keV * u.cm**2 * u.s)
spectrum.K.bounds  = (1e-35, 1e-10) / (u.TeV * u.cm**2 * u.s)
spectrum.alpha        = -2.5
spectrum.alpha.bounds = (-4., 2.)
spectrum.beta         = 0.0
spectrum.beta.bounds  = (-4., 2.)

model = Model(source)
data  = DataList(hawc)

jl = JointLikelihood(model, data, verbose=False)


logL_cache = {}
# ── Likelihood evaluator ──────────────────────────────────────────────────────
def evaluate_logL(log_k: float, alpha: float, beta: float) -> float:
    key = (round(log_k, 6), round(alpha, 6), round(beta, 6))
    if key in logL_cache:
        print("  [cache hit]")
        return logL_cache[key]
    try:
        spectrum.K     = 10**log_k / (u.keV * u.cm**2 * u.s)
        spectrum.alpha = alpha
        spectrum.beta  = beta
        logL = hawc.get_log_like()
        logL_cache[key] = logL
        return logL
    except Exception as e:
        print(f"  [logL eval error] {e}")
        logL_cache[key] = -1e9
        return -1e9

# ── Seed ──────────────────────────────────────────────────────────────────────
SEED_LOG = ""


def parse_seed(log_text: str) -> list:
    pattern = r"trial values:\s*([\-\d.e+]+),([\-\d.e+]+),([\-\d.e+]+)\s*->\s*logL\s*=\s*([\-\d.]+)"
    trials = []
    for m in re.finditer(pattern, log_text):
        log_k, alpha, beta, logL = map(float, m.groups())
        trials.append({"log_k": log_k, "alpha": alpha, "beta": beta, "logL": logL})
    return trials


# Raised from -1e8: excludes bad probes like -1242386 while keeping all
# reasonable evaluations (good region sits around -18000 to -25000)(need to readjust if we change the model or data drastically)
LOGLIKE_FLOOR = -5e5  #−500000


# ── Gradient estimation ───────────────────────────────────────────────────────
def estimate_gradient(history: list, best: dict, eps: dict) -> dict:
    clean = [t for t in history if t["logL"] > LOGLIKE_FLOOR]
    grad_raw = {}
    for param, step in eps.items():
        neighbors = [
            t for t in clean
            if abs(t[param] - best[param]) > 0
            and all(abs(t[p] - best[p]) < step * 3 for p in eps if p != param)
        ]
        if neighbors:
            neighbors.sort(key=lambda t: abs(t[param] - best[param]))
            n = neighbors[0]
            grad_raw[param] = (n["logL"] - best["logL"]) / (n[param] - best[param])
        else:
            grad_raw[param] = 0.0
    max_abs = max(abs(v) for v in grad_raw.values()) or 1.0
    return {k: v / max_abs for k, v in grad_raw.items()}


# ── Robust JSON extraction ────────────────────────────────────────────────────
def parse_llm_response(text: str) -> dict | None:
    match = re.search(r'\{[^}]*\}', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if all(k in parsed for k in ("log_k", "alpha", "beta")):
                return parsed
        except json.JSONDecodeError:
            pass
    lk = re.search(r'log_k["\s]*:\s*([-\d.e+]+)', text)
    al = re.search(r'alpha["\s]*:\s*([-\d.e+]+)', text)
    bt = re.search(r'beta["\s]*:\s*([-\d.e+]+)', text)
    if lk and al and bt:
        print("  [JSON repair] extracted values via regex fallback")
        return {"log_k": float(lk.group(1)), "alpha": float(al.group(1)), "beta": float(bt.group(1))}
    return None


# ── Stall detection ───────────────────────────────────────────────────────────
def is_stalled(history: list, n: int = 3, tol: float = 0.05) -> bool:
    if len(history) < n:
        return False
    recent = history[-n:]
    return all(
        max(t[p] for t in recent) - min(t[p] for t in recent) <= tol
        for p in ("log_k", "alpha", "beta")
    )


# ── LLM proposal ─────────────────────────────────────────────────────────────
def llm_propose(history: list) -> dict:
    best = max(history, key=lambda x: x["logL"])
    eps  = {"log_k": 2.0, "alpha": 0.5, "beta": 0.25}
    grad = estimate_gradient(history, best, eps)
    print(f"  [grad] log_k={grad['log_k']:+.3f}  alpha={grad['alpha']:+.3f}  beta={grad['beta']:+.3f}")

    clean = [t for t in history if t["logL"] > LOGLIKE_FLOOR]

    # Top-10 by logL: shows the LLM the best-explored region
    top10 = sorted(clean, key=lambda x: x["logL"], reverse=True)[:10]
    top_str = "\n".join(
        f"  log10(K)={t['log_k']:.4f}  alpha={t['alpha']:.4f}  beta={t['beta']:.5f}  logL={t['logL']:.3f}"
        for t in top10
    )

    # Last 5 chronological: shows recent movement direction
    recent5 = clean[-5:]
    recent_str = "\n".join(
        f"  log10(K)={t['log_k']:.4f}  alpha={t['alpha']:.4f}  beta={t['beta']:.5f}  logL={t['logL']:.3f}"
        for t in recent5
    )

    prompt = f"""You are acting as the Minuit MIGRAD optimizer for a 3-parameter log-parabola gamma-ray spectral fit.
GOAL: MAXIMIZE logL (less negative = better).

Parameters and hard bounds:
  log10(K) in [-44, -19]   — flux normalization (K in keV^-1 cm^-2 s^-1)
  alpha    in [-4,   2]    — spectral index
  beta     in [-4,   2]    — spectral curvature

Current best point:
  log10(K)={best['log_k']:.4f}  alpha={best['alpha']:.4f}  beta={best['beta']:.5f}  logL={best['logL']:.3f}

Normalized gradient direction at current best (scale: -1 to +1, positive = increase that param):
  ∂logL/∂log10(K) = {grad['log_k']:+.3f}   {'→ INCREASE log10(K)' if grad['log_k'] > 0 else '→ DECREASE log10(K)' if grad['log_k'] < 0 else '→ gradient unclear'}
  ∂logL/∂alpha    = {grad['alpha']:+.3f}   {'→ INCREASE alpha' if grad['alpha'] > 0 else '→ DECREASE alpha' if grad['alpha'] < 0 else '→ gradient unclear'}
  ∂logL/∂beta     = {grad['beta']:+.3f}   {'→ INCREASE beta' if grad['beta'] > 0 else '→ DECREASE beta' if grad['beta'] < 0 else '→ gradient unclear'}

You MUST update ALL THREE parameters simultaneously in every proposal.
Never hold any parameter fixed — treat each proposal as a direction vector
in 3D space (log_k, alpha, beta), not three separate 1D steps.
Move each parameter in its gradient direction; scale the move size down
if recent proposals have not improved logL.

Top-10 evaluated points (best first):
{top_str}

Last 5 evaluated points (newest last):
{recent_str}

Reply with JSON only — no explanation, no markdown:
{{"log_k": <float>, "alpha": <float>, "beta": <float>}}"""

    try:
        resp = client.chat.completions.create(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.3,
        )
        text = resp.choices[0].message.content.strip()
        parsed = parse_llm_response(text)
        if parsed:
            return parsed
        print(f"  [LLM warning] could not extract params from: {text[:80]}")
    except Exception as e:
        print(f"  [LLM error] {e}")

    return {
        "log_k": best["log_k"] + random.gauss(0, 0.05),
        "alpha": best["alpha"] + random.gauss(0, 0.05),
        "beta":  best["beta"]  + random.gauss(0, 0.02),
    }


def clip(proposal: dict) -> dict:
    proposal["log_k"] = max(-44.0, min(-19.0, proposal["log_k"]))
    proposal["alpha"] = max( -4.0, min(  2.0, proposal["alpha"]))
    proposal["beta"]  = max( -4.0, min(  2.0, proposal["beta"]))
    return proposal


def dedup(proposal: dict, max_tries: int = 8) -> dict:
    for attempt in range(max_tries):
        key = (round(proposal["log_k"], 6), round(proposal["alpha"], 6), round(proposal["beta"], 6))
        if key not in logL_cache:
            return proposal
        scale = 0.3 * (2 ** attempt)  # double perturbation size each retry
        proposal = clip({
            "log_k": proposal["log_k"] + random.gauss(0, scale),
            "alpha": proposal["alpha"] + random.gauss(0, scale * 0.3),
            "beta":  proposal["beta"]  + random.gauss(0, scale * 0.1),
        })
        print(f"  [dedup retry {attempt+1}, scale={scale:.2f}]")
    return proposal


# ── Main ──────────────────────────────────────────────────────────────────────
N_ITERATIONS = 200

history = parse_seed(SEED_LOG)

if not history:
    print("No seed data — probing each parameter (Minuit finite-difference init)...")
    probes = [
        (-23.0, -2.5,  0.0),
        (-20.0, -2.5,  0.0),
        (-26.0, -2.5,  0.0),
        (-23.0, -3.5,  0.0),
        (-23.0, -1.5,  0.0),
        (-23.0, -2.5, -0.5),
        (-23.0, -2.5,  0.5),
        (-20.0, -3.5,  0.5),
        (-20.0, -1.5, -0.5),
    ]
    for lk, a, b in probes:
        logL = evaluate_logL(lk, a, b)
        history.append({"log_k": lk, "alpha": a, "beta": b, "logL": logL})
        print(f"  probe  log10(K)={lk:.1f}  alpha={a:.1f}  beta={b:.1f}  logL={logL:.3f}")

seed_best = max(history, key=lambda x: x["logL"])
print(f"Seed: {len(history)} points | best logL = {seed_best['logL']:.3f}")
print(f"Running {N_ITERATIONS} OPRO iterations via Ollama (llama3.1:8b)...\n")

for i in range(N_ITERATIONS):
    if is_stalled(history):
        best_pt = max(history, key=lambda x: x["logL"])
        proposal = dedup(clip({
            "log_k": best_pt["log_k"] + random.gauss(0, 0.5),
            "alpha": best_pt["alpha"] + random.gauss(0, 0.3),
            "beta":  best_pt["beta"]  + random.gauss(0, 0.1),
        }))
        print("  [stall — exploring]")
    else:
        proposal = dedup(clip(llm_propose(history)))

    proposal["logL"] = evaluate_logL(**{k: v for k, v in proposal.items() if k != "logL"})
    history.append(proposal)

    best = max(history, key=lambda x: x["logL"])
    print(
        f"[{i+1:3d}/{N_ITERATIONS}]  "
        f"log10(K)={proposal['log_k']:.4f}  alpha={proposal['alpha']:.4f}  beta={proposal['beta']:.5f}  "
        f"logL={proposal['logL']:.3f}  |  best={best['logL']:.3f}"
    )

# ── OPRO Results ──────────────────────────────────────────────────────────────
best = max(history, key=lambda x: x["logL"])

print("\n=== OPRO Best Fit ===")
print(f"log10(K) = {best['log_k']:.4f}  ->  K = {10**best['log_k']:.4e} keV^-1 cm^-2 s^-1")
print(f"alpha    = {best['alpha']:.4f}")
print(f"beta     = {best['beta']:.5f}")
print(f"logL     = {best['logL']:.3f}")

# ── Minuit warm-start from OPRO best-fit (for uncertainties) ──────────────────
print("\nRunning Minuit from OPRO best-fit to obtain uncertainties...")
spectrum.K     = 10**best["log_k"] / (u.keV * u.cm**2 * u.s)
spectrum.alpha = best["alpha"]
spectrum.beta  = best["beta"]

fit_results, like_df = jl.fit(quiet=True)

print("\n=== Final Fit (OPRO warm-start + Minuit HESSE) ===")
print(fit_results)

fig = hawc.display_spectrum()
fig.savefig("opro_crab_residuals.png")

fig = hawc.display_fit(smoothing_kernel_sigma=0.3, display_colorbar=True)
fig.savefig("opro_crab_fit_planes.png")

print("\nPlots saved: opro_crab_residuals.png, opro_crab_fit_planes.png")
