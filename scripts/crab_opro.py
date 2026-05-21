from hawc_hal import HAL, HealpixConeROI
import matplotlib.pyplot as plt
from threeML import *
from openai import OpenAI
import json
import re
import os

# ── GROQ client ──────────────────────────────────────────────────────────────
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

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
spectrum.K         = 1e-14 / (u.TeV * u.cm**2 * u.s)
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
"""
SEED_LOG = """
trial values: -23,-2.5,-2.8189e-17 -> logL = -23541.653
trial values: -22.951,-2.5,-2.8189e-17 -> logL = -23453.068
trial values: -23.049,-2.5,-2.8189e-17 -> logL = -23623.529
trial values: -22.995,-2.5,-2.8189e-17 -> logL = -23533.102
trial values: -23.005,-2.5,-2.8189e-17 -> logL = -23550.137
trial values: -22.364,-2.5,-2.8189e-17 -> logL = -21676.379
trial values: -21.774,-2.5,-2.8189e-17 -> logL = -18997.740
trial values: -21.773,-2.5,-2.8189e-17 -> logL = -18993.470
trial values: -21.763,-2.5222,-0.0083074 -> logL = -18939.497
trial values: -21.739,-2.6023,-0.014562 -> logL = -18852.858
trial values: -21.679,-2.614,-0.0002383 -> logL = -18805.751
trial values: -21.671,-2.6154,0.0015549 -> logL = -18805.126
trial values: -21.648,-2.6084,0.024298 -> logL = -18786.364
trial values: -21.619,-2.6,0.051251 -> logL = -18775.576
trial values: -21.609,-2.5969,0.060987 -> logL = -18774.121
trial values: -21.603,-2.595,0.06698 -> logL = -18773.801
trial values: -21.606,-2.6346,0.083751 -> logL = -18766.901
trial values: -21.607,-2.6457,0.088508 -> logL = -18766.528
trial values: -21.600,-2.6465,0.096266 -> logL = -18766.019
trial values: -21.595,-2.6471,0.1024 -> logL = -18765.887
trial values: -21.595,-2.6455,0.10337 -> logL = -18765.837
"""
"""

def parse_seed(log_text: str) -> list:
    pattern = r"trial values:\s*([\-\d.e+]+),([\-\d.e+]+),([\-\d.e+]+)\s*->\s*logL\s*=\s*([\-\d.]+)"
    trials = []
    for m in re.finditer(pattern, log_text):
        log_k, alpha, beta, logL = map(float, m.groups())
        trials.append({"log_k": log_k, "alpha": alpha, "beta": beta, "logL": logL})
    return trials


# ── OPRO: LLM proposes next parameters ───────────────────────────────────────
def llm_propose(history: list, n_best: int = 15) -> dict:
    best = sorted(history, key=lambda x: x["logL"], reverse=True)[:n_best]
    history_str = "\n".join(
        f"  log10(K)={t['log_k']:.4f}, alpha={t['alpha']:.4f}, beta={t['beta']:.5f} -> logL={t['logL']:.3f}"
        for t in best
    )

    prompt = f"""You are optimizing a 3-parameter log-parabola gamma-ray spectral fit to HAWC telescope data.
GOAL: MAXIMIZE logL (the value should become LESS negative).

Parameters and bounds:
  log10(K) in [-35, -10]   — overall flux normalization (log scale)
  alpha    in [-4,   2]    — spectral index
  beta     in [-4,   2]    — spectral curvature

Known correlations (use these to guide proposals):
  log10(K) and beta are STRONGLY positively correlated (r = +0.83)
  alpha    and beta are moderately anti-correlated     (r = -0.43)

Top {len(best)} trials so far (best first):
{history_str}

Current best logL = {best[0]['logL']:.3f}.
Analyze the trend and propose the next parameter set to evaluate.
Reply with JSON only — no explanation, no markdown:
{{"log_k": <float>, "alpha": <float>, "beta": <float>}}"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=1,
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
    bp = best[0]
    return {
        "log_k": bp["log_k"] + random.gauss(0, 0.05),
        "alpha": bp["alpha"] + random.gauss(0, 0.05),
        "beta":  bp["beta"]  + random.gauss(0, 0.02),
    }


def clip(proposal: dict) -> dict:
    proposal["log_k"] = max(-35.0, min(-10.0, proposal["log_k"]))
    proposal["alpha"] = max( -4.0, min(  2.0, proposal["alpha"]))
    proposal["beta"]  = max( -4.0, min(  2.0, proposal["beta"]))
    return proposal


# ── Main ──────────────────────────────────────────────────────────────────────
N_ITERATIONS = 30

history = parse_seed(SEED_LOG)

if not history:
    print("No seed data — evaluating cold-start point...")
    #logL0 = evaluate_logL(-23.0, -2.5, 0.0)
    logL0 = evaluate_logL(-15.0, -0.0, 0.0)
    history.append({"log_k": -23.0, "alpha": -2.5, "beta": 0.0, "logL": logL0})

seed_best = max(history, key=lambda x: x["logL"])
print(f"Seed: {len(history)} points | best logL = {seed_best['logL']:.3f}")
print(f"Running {N_ITERATIONS} OPRO iterations via Groq...\n")

for i in range(N_ITERATIONS):
    proposal = clip(llm_propose(history, n_best=min(15, len(history))))
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
print(f"log10(K) = {best['log_k']:.4f}  ->  K = {10**best['log_k']:.4e} TeV^-1 cm^-2 s^-1")
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
