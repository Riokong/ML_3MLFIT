#!/usr/bin/env python3
"""Export Crab fit inputs and sampled likelihood evaluations for surrogate modeling.

This script extracts two kinds of data from the same HAL/threeML setup used in
`crab.py`:

1. Static fit inputs:
   - active bin ids
   - active HEALPix pixel ids
   - observed counts per active pixel
   - background counts per active pixel
   - per-bin metadata such as number of transits

2. Sampled likelihood evaluations:
   - trial values of (K, alpha, beta)
   - the resulting total minus log-likelihood
   - per-bin minus log-likelihood contributions
   - per-bin model-count sums
   - optionally the full per-pixel model vectors used in each evaluation

The ROOT minimizer never sees the HAWC maps directly. It sees only a parameter
vector and calls back into threeML/HAL to evaluate the objective. This exporter
captures the fixed arrays and the objective evaluations so you can train a
surrogate model for the fit.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from hawc_hal import HAL, HealpixConeROI
from hawc_hal.log_likelihood import log_likelihood
from threeML import DataList, JointLikelihood, Log_parabola, Model, PointSource, u


RA_CRAB = 83.63
DEC_CRAB = 22.02
DATA_RADIUS_DEG = 3.0
MODEL_RADIUS_DEG = 8.0
MAPTREE = "HAWC_9bin_507days_crab_data.hd5"
RESPONSE = "HAWC_9bin_507days_crab_response.hd5"

# Same seed values and bounds as crab.py.
DEFAULT_K_TEv = 1e-14
DEFAULT_LOG10_K_BOUNDS = (-35.0, -10.0)
DEFAULT_ALPHA_BOUNDS = (-4.0, 2.0)
DEFAULT_BETA_BOUNDS = (-4.0, 2.0)


def build_analysis() -> tuple[HAL, object]:
    roi = HealpixConeROI(
        data_radius=DATA_RADIUS_DEG,
        model_radius=MODEL_RADIUS_DEG,
        ra=RA_CRAB,
        dec=DEC_CRAB,
    )

    hawc = HAL(
        "HAWC",
        MAPTREE,
        RESPONSE,
        roi,
        flat_sky_pixels_size=0.1,
    )
    hawc.set_active_measurements(1, 9)

    spectrum = Log_parabola()
    source = PointSource("crab", ra=RA_CRAB, dec=DEC_CRAB, spectral_shape=spectrum)
    spectrum.piv = 7 * u.TeV
    spectrum.piv.fix = True
    spectrum.K = DEFAULT_K_TEv / (u.TeV * u.cm**2 * u.s)
    spectrum.K.bounds = (
        10 ** DEFAULT_LOG10_K_BOUNDS[0],
        10 ** DEFAULT_LOG10_K_BOUNDS[1],
    ) / (u.TeV * u.cm**2 * u.s)
    spectrum.alpha = -2.5
    spectrum.alpha.bounds = DEFAULT_ALPHA_BOUNDS
    spectrum.beta = 0.0
    spectrum.beta.bounds = DEFAULT_BETA_BOUNDS

    model = Model(source)

    # Build the same object graph as crab.py so HAL has the model attached.
    JointLikelihood(model, DataList(hawc), verbose=False)

    return hawc, spectrum


def flatten_per_bin(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.array([arr.shape[0] for arr in arrays], dtype=np.int64)
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    flat = np.concatenate(arrays) if arrays else np.array([], dtype=float)
    return flat, offsets


def extract_static_inputs(hawc: HAL) -> dict[str, np.ndarray]:
    obs_per_bin = []
    bkg_per_bin = []
    pix_per_bin = []
    n_transits = []
    log_factorials = []
    saturated_bias = []

    for bin_id in hawc._active_planes:
        data_bin = hawc._maptree[bin_id]
        obs_per_bin.append(np.asarray(data_bin.observation_map.as_partial(), dtype=np.float64))
        bkg_per_bin.append(np.asarray(data_bin.background_map.as_partial(), dtype=np.float64))
        pix_per_bin.append(np.asarray(hawc._active_pixels[bin_id], dtype=np.int64))
        n_transits.append(data_bin.n_transits)
        log_factorials.append(hawc._log_factorials[bin_id])
        saturated_bias.append(hawc._saturated_model_like_per_maptree[bin_id])

    obs_flat, offsets = flatten_per_bin(obs_per_bin)
    bkg_flat, _ = flatten_per_bin(bkg_per_bin)
    pix_flat, _ = flatten_per_bin(pix_per_bin)

    return {
        "bin_ids": np.array(hawc._active_planes),
        "offsets": offsets,
        "active_pixel_ids_flat": pix_flat,
        "obs_counts_flat": obs_flat,
        "bkg_counts_flat": bkg_flat,
        "n_transits": np.asarray(n_transits, dtype=np.float64),
        "log_factorials": np.asarray(log_factorials, dtype=np.float64),
        "saturated_model_bias": np.asarray(saturated_bias, dtype=np.float64),
    }


def draw_parameter_samples(
    n_samples: int,
    seed: int,
    log10_k_bounds: tuple[float, float],
    alpha_bounds: tuple[float, float],
    beta_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    log10_k = rng.uniform(log10_k_bounds[0], log10_k_bounds[1], size=n_samples)
    alpha = rng.uniform(alpha_bounds[0], alpha_bounds[1], size=n_samples)
    beta = rng.uniform(beta_bounds[0], beta_bounds[1], size=n_samples)

    # Ensure the exact seed point from crab.py is included as sample 0.
    if n_samples > 0:
        log10_k[0] = np.log10(DEFAULT_K_TEv)
        alpha[0] = -2.5
        beta[0] = 0.0

    return log10_k, alpha, beta


def evaluate_samples(
    hawc: HAL,
    spectrum,
    log10_k: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    store_model_maps: bool,
) -> dict[str, np.ndarray]:
    n_samples = log10_k.shape[0]
    n_bins = len(hawc._active_planes)

    total_minus_log_like = np.zeros(n_samples, dtype=np.float64)
    per_bin_minus_log_like = np.zeros((n_samples, n_bins), dtype=np.float64)
    per_bin_model_sum = np.zeros((n_samples, n_bins), dtype=np.float64)
    model_maps = []

    for i in range(n_samples):
        spectrum.K = (10 ** log10_k[i]) / (u.TeV * u.cm**2 * u.s)
        spectrum.alpha = float(alpha[i])
        spectrum.beta = float(beta[i])

        n_point_sources = hawc._likelihood_model.get_number_of_point_sources()
        n_ext_sources = hawc._likelihood_model.get_number_of_extended_sources()
        bkg_renorm = list(hawc.nuisance_parameters.values())[0].value

        total_log_like = 0.0
        sample_model_maps = []

        for j, bin_id in enumerate(hawc._active_planes):
            data_bin = hawc._maptree[bin_id]

            obs = np.asarray(data_bin.observation_map.as_partial(), dtype=np.float64)
            bkg = np.asarray(data_bin.background_map.as_partial(), dtype=np.float64) * bkg_renorm
            mdl = np.asarray(
                hawc._get_expectation(data_bin, bin_id, n_point_sources, n_ext_sources),
                dtype=np.float64,
            )

            pseudo_log_like = log_likelihood(obs, bkg, mdl)
            this_log_like = (
                pseudo_log_like
                - hawc._log_factorials[bin_id]
                - hawc._saturated_model_like_per_maptree[bin_id]
            )

            total_log_like += this_log_like
            per_bin_minus_log_like[i, j] = -this_log_like
            per_bin_model_sum[i, j] = mdl.sum()

            if store_model_maps:
                sample_model_maps.append(mdl)

        total_minus_log_like[i] = -total_log_like

        if store_model_maps:
            model_maps.append(np.concatenate(sample_model_maps))

    results = {
        "log10_k": log10_k,
        "alpha": alpha,
        "beta": beta,
        "minus_log_like": total_minus_log_like,
        "per_bin_minus_log_like": per_bin_minus_log_like,
        "per_bin_model_sum": per_bin_model_sum,
    }

    if store_model_maps:
        results["model_counts_flat"] = np.stack(model_maps, axis=0)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=256, help="Number of trial parameter points to evaluate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for parameter sampling.")
    parser.add_argument(
        "--output-prefix",
        default="crab_training",
        help="Prefix for output .npz files written in the working directory.",
    )
    parser.add_argument(
        "--store-model-maps",
        action="store_true",
        help="Store the full trial-dependent model vector for every sample.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write extracted data to CSV files.",
    )
    parser.add_argument("--log10-k-min", type=float, default=DEFAULT_LOG10_K_BOUNDS[0])
    parser.add_argument("--log10-k-max", type=float, default=DEFAULT_LOG10_K_BOUNDS[1])
    parser.add_argument("--alpha-min", type=float, default=DEFAULT_ALPHA_BOUNDS[0])
    parser.add_argument("--alpha-max", type=float, default=DEFAULT_ALPHA_BOUNDS[1])
    parser.add_argument("--beta-min", type=float, default=DEFAULT_BETA_BOUNDS[0])
    parser.add_argument("--beta-max", type=float, default=DEFAULT_BETA_BOUNDS[1])
    return parser.parse_args()


def write_static_csv(prefix: Path, static_inputs: dict[str, np.ndarray]) -> tuple[Path, Path]:
    pixels_path = prefix.with_name(f"{prefix.name}_static_pixels.csv")
    bins_path = prefix.with_name(f"{prefix.name}_static_bins.csv")

    bin_ids = static_inputs["bin_ids"]
    offsets = static_inputs["offsets"]
    active_pixel_ids = static_inputs["active_pixel_ids_flat"]
    obs_counts = static_inputs["obs_counts_flat"]
    bkg_counts = static_inputs["bkg_counts_flat"]
    n_transits = static_inputs["n_transits"]
    log_factorials = static_inputs["log_factorials"]
    saturated_model_bias = static_inputs["saturated_model_bias"]

    with pixels_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["bin_id", "pixel_index_in_bin", "active_pixel_id", "obs_counts", "bkg_counts"]
        )
        for i, bin_id in enumerate(bin_ids):
            start = int(offsets[i])
            stop = int(offsets[i + 1])
            for j in range(start, stop):
                writer.writerow(
                    [
                        bin_id,
                        j - start,
                        int(active_pixel_ids[j]),
                        float(obs_counts[j]),
                        float(bkg_counts[j]),
                    ]
                )

    with bins_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "bin_id",
                "n_pixels",
                "n_transits",
                "log_factorial",
                "saturated_model_bias",
            ]
        )
        for i, bin_id in enumerate(bin_ids):
            writer.writerow(
                [
                    bin_id,
                    int(offsets[i + 1] - offsets[i]),
                    float(n_transits[i]),
                    float(log_factorials[i]),
                    float(saturated_model_bias[i]),
                ]
            )

    return pixels_path, bins_path


def write_samples_csv(prefix: Path, static_inputs: dict[str, np.ndarray], sampled: dict[str, np.ndarray]) -> Path:
    samples_path = prefix.with_name(f"{prefix.name}_samples.csv")
    bin_ids = [str(x) for x in static_inputs["bin_ids"]]

    header = ["sample_id", "log10_k", "alpha", "beta", "minus_log_like"]
    header.extend([f"minus_log_like_bin_{bin_id}" for bin_id in bin_ids])
    header.extend([f"model_sum_bin_{bin_id}" for bin_id in bin_ids])

    with samples_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(sampled["minus_log_like"].shape[0]):
            row = [
                i,
                float(sampled["log10_k"][i]),
                float(sampled["alpha"][i]),
                float(sampled["beta"][i]),
                float(sampled["minus_log_like"][i]),
            ]
            row.extend(float(x) for x in sampled["per_bin_minus_log_like"][i])
            row.extend(float(x) for x in sampled["per_bin_model_sum"][i])
            writer.writerow(row)

    return samples_path


def main() -> None:
    args = parse_args()
    hawc, spectrum = build_analysis()

    static_inputs = extract_static_inputs(hawc)
    log10_k, alpha, beta = draw_parameter_samples(
        n_samples=args.n_samples,
        seed=args.seed,
        log10_k_bounds=(args.log10_k_min, args.log10_k_max),
        alpha_bounds=(args.alpha_min, args.alpha_max),
        beta_bounds=(args.beta_min, args.beta_max),
    )
    sampled = evaluate_samples(
        hawc=hawc,
        spectrum=spectrum,
        log10_k=log10_k,
        alpha=alpha,
        beta=beta,
        store_model_maps=args.store_model_maps,
    )

    prefix = Path(args.output_prefix)
    static_path = prefix.with_name(f"{prefix.name}_static.npz")
    samples_path = prefix.with_name(f"{prefix.name}_samples.npz")

    np.savez_compressed(static_path, **static_inputs)
    np.savez_compressed(samples_path, **sampled)

    print(f"wrote {static_path}")
    print(f"wrote {samples_path}")
    print(f"active bins: {list(static_inputs['bin_ids'])}")
    print(f"active data points: {static_inputs['obs_counts_flat'].shape[0]}")
    print(f"samples: {args.n_samples}")
    if args.store_model_maps:
        print("stored full trial model maps")
    if args.write_csv:
        pixels_csv, bins_csv = write_static_csv(prefix, static_inputs)
        samples_csv = write_samples_csv(prefix, static_inputs, sampled)
        print(f"wrote {pixels_csv}")
        print(f"wrote {bins_csv}")
        print(f"wrote {samples_csv}")


if __name__ == "__main__":
    main()
