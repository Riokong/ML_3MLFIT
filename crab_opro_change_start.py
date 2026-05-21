# cold start matches crab.py initialization: log_k=-23.0 (K=1e-14 TeV^-1), alpha=-2.5, beta=0.0

from hawc_hal import HAL, HealpixConeROI
import matplotlib.pyplot as plt
from threeML import *
from groq import Groq
import json
import re
import os

# ── GROQ client ──────────────────────────────────────────────────────────────
client = Groq()

# ── HAWC setup (identical to crab.py) ────────────────────────────────────────
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

# Links the model to the HAL plugin so get_log_like() works without calling fit()
jl = JointLikelihood(model, data, verbose=False)

# ── Likelihood evaluator (uses real HAWC data — no approximation) ─────────────
def evaluate_logL(log_k: float, alpha: float, beta: float) -> float:
    # log_k is log10(K) where K is in keV^-1 cm^-2 s^-1 (threeML internal units)
    try:
        spectrum.K     = 10**log_k / (u.keV * u.cm**2 * u.s)
        spectrum.alpha = alpha
        spectrum.beta  = beta
        return hawc.get_log_like()
    except Exception as e:
        print(f"  [logL eval error] {e}")
        return -1e9

# ── Seed: Minuit trial log from crab.py ────────────────────────────────────
# Leave empty ("") to start cold (one initial evaluation will be done).

SEED_LOG = ""


def parse_seed(log_text: str) -> list:
    pattern = r"trial values:\s*([\-\d.e+]+),([\-\d.e+]+),([\-\d.e+]+)\s*->\s*logL\s*=\s*([\-\d.]+)"
    trials = []
    for m in re.finditer(pattern, log_text):
        log_k, alpha, beta, logL = map(float, m.groups())
        trials.append({"log_k": log_k, "alpha": alpha, "beta": beta, "logL": logL})
    return trials


LOGLIKE_FLOOR = -1e8  # filter unphysical evaluations from gradient computation

# ── Gradient estimation from nearby points ────────────────────────────────────
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
    # Normalize to [-1, 1] so the LLM sees consistent scale regardless of logL magnitude
    max_abs = max(abs(v) for v in grad_raw.values()) or 1.0
    return {k: v / max_abs for k, v in grad_raw.items()}


# ── LLM-as-Minuit: proposes next parameters using pre-computed gradient ────────
def llm_propose(history: list, n_best: int = 15) -> dict:
    best = max(history, key=lambda x: x["logL"])
    eps = {"log_k": 2.0, "alpha": 0.5, "beta": 0.25}
    grad = estimate_gradient(history, best, eps)
    print(f"  [grad] log_k={grad['log_k']:+.3f}  alpha={grad['alpha']:+.3f}  beta={grad['beta']:+.3f}")

    history_str = "\n".join(
        f"  step={i+1}  log10(K)={t['log_k']:.4f}  alpha={t['alpha']:.4f}  beta={t['beta']:.5f}  logL={t['logL']:.3f}"
        for i, t in enumerate(history[-20:], start=max(0, len(history)-20))
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

Recommended step sizes per iteration (use these as a guide — scale down if recent steps are not improving):
  log10(K): ±0.3   alpha: ±0.05   beta: ±0.02
Move in the gradient direction. Do NOT jump more than 2x the recommended step size.
Recent evaluated points (newest last):
{history_str}

Reply with JSON only — no explanation, no markdown:
{{"log_k": <float>, "alpha": <float>, "beta": <float>}}"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.3,
        )
        text = resp.choices[0].message.content.strip()
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            if all(k in parsed for k in ("log_k", "alpha", "beta")):
                return parsed
            print(f"  [LLM warning] unexpected keys {list(parsed.keys())} — perturbing best point")
    except Exception as e:
        print(f"  [LLM error] {e} — perturbing best point")

    # Fallback: small random perturbation around current best
    import random
    bp = best
    return {
        "log_k": bp["log_k"] + random.gauss(0, 0.05),
        "alpha": bp["alpha"] + random.gauss(0, 0.05),
        "beta":  bp["beta"]  + random.gauss(0, 0.02),
    }


def clip(proposal: dict) -> dict:
    proposal["log_k"] = max(-44.0, min(-19.0, proposal["log_k"]))
    proposal["alpha"] = max( -4.0, min(  2.0, proposal["alpha"]))
    proposal["beta"]  = max( -4.0, min(  2.0, proposal["beta"]))
    return proposal


# ── Main ──────────────────────────────────────────────────────────────────────
N_ITERATIONS = 50

history = parse_seed(SEED_LOG)

if not history:
    print("No seed data — probing each parameter (Minuit finite-difference init)...")
    probes = [
        (-23.0, -2.5,  0.0),   # base (matches crab.py initialization)
        (-20.0, -2.5,  0.0),   # log_k up
        (-26.0, -2.5,  0.0),   # log_k down
        (-23.0, -3.5,  0.0),   # alpha negative
        (-23.0, -1.5,  0.0),   # alpha positive
        (-23.0, -2.5, -0.5),   # beta negative
        (-23.0, -2.5,  0.5),   # beta positive
        (-20.0, -3.5,  0.5),   # diagonal: log_k up + alpha down + beta up
        (-20.0, -1.5, -0.5),   # diagonal: log_k up + alpha up + beta down
    ]
    for lk, a, b in probes:
        logL = evaluate_logL(lk, a, b)
        history.append({"log_k": lk, "alpha": a, "beta": b, "logL": logL})
        print(f"  probe  log10(K)={lk:.1f}  alpha={a:.1f}  beta={b:.1f}  logL={logL:.3f}")

seed_best = max(history, key=lambda x: x["logL"])
print(f"Seed: {len(history)} points | best logL = {seed_best['logL']:.3f}")
print(f"Running {N_ITERATIONS} OPRO iterations via Groq...\n")

for i in range(N_ITERATIONS):
    proposal = clip(llm_propose(history))
    proposal["logL"] = evaluate_logL(**{k: v for k, v in proposal.items() if k != "logL"})
    history.append(proposal)

    best = max(history, key=lambda x: x["logL"])
    print(
        f"[{i+1:3d}/{N_ITERATIONS}]  "
        f"log10(K)={proposal['log_k']:.4f}  alpha={proposal['alpha']:.4f}  beta={proposal['beta']:.5f}  "
        f"logL={proposal['logL']:.3f}  |  best={best['logL']:.3f}"
    )

# ── Results ───────────────────────────────────────────────────────────────────
best = max(history, key=lambda x: x["logL"])

print("\n=== OPRO Best Fit ===")
print(f"log10(K) = {best['log_k']:.4f}  ->  K = {10**best['log_k']:.4e} keV^-1 cm^-2 s^-1")
print(f"alpha    = {best['alpha']:.4f}")
print(f"beta     = {best['beta']:.5f}")
print(f"logL     = {best['logL']:.3f}")
print(f"(Minuit: logL = -18765.837)")

# Apply best parameters and produce standard threeML plots
spectrum.K     = 10**best["log_k"] / (u.keV * u.cm**2 * u.s)
spectrum.alpha = best["alpha"]
spectrum.beta  = best["beta"]

fig = hawc.display_spectrum()
fig.savefig("opro_crab_residuals.png")

fig = hawc.display_fit(smoothing_kernel_sigma=0.3, display_colorbar=True)
fig.savefig("opro_crab_fit_planes.png")

print("\nPlots saved: opro_crab_residuals.png, opro_crab_fit_planes.png")
