"""Grey Wolf Optimizer (GWO) for Autoencoder Hyperparameter Tuning."""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Tuple
import numpy as np

from training.metaheuristics.search_space import (
    DIMENSION,
    Hyperparameters,
    decode_vector,
)

logger = logging.getLogger("gwo")


def run_gwo(
    eval_fn: Callable[[Hyperparameters], Tuple[float, Dict]],
    n_wolves: int = 6,
    n_iterations: int = 5,
    seed: int = 42,
) -> Dict:
    """Run Grey Wolf Optimizer over the continuous [0, 1]^D search space."""
    rng = np.random.default_rng(seed)
    dim = DIMENSION

    # Initialize positions in [0, 1]
    positions = rng.uniform(0.05, 0.95, size=(n_wolves, dim))
    fitnesses = np.full(n_wolves, -float("inf"))

    eval_history: List[Dict] = []
    iteration_history: List[Dict] = []

    alpha_pos = np.zeros(dim)
    alpha_fitness = -float("inf")
    alpha_hp = None

    beta_pos = np.zeros(dim)
    beta_fitness = -float("inf")

    delta_pos = np.zeros(dim)
    delta_fitness = -float("inf")

    eval_count = 0
    start_time = time.perf_counter()

    logger.info("Initializing GWO pack (N=%d, T=%d, Dim=%d)...", n_wolves, n_iterations, dim)

    # Initial evaluation
    for i in range(n_wolves):
        eval_count += 1
        pos = positions[i]
        hp = decode_vector(pos)
        fitness, info = eval_fn(hp)
        fitnesses[i] = fitness

        # Update Alpha, Beta, Delta
        if fitness > alpha_fitness:
            delta_pos = beta_pos.copy()
            delta_fitness = beta_fitness
            beta_pos = alpha_pos.copy()
            beta_fitness = alpha_fitness
            alpha_pos = pos.copy()
            alpha_fitness = fitness
            alpha_hp = hp
        elif fitness > beta_fitness:
            delta_pos = beta_pos.copy()
            delta_fitness = beta_fitness
            beta_pos = pos.copy()
            beta_fitness = fitness
        elif fitness > delta_fitness:
            delta_pos = pos.copy()
            delta_fitness = fitness

        eval_history.append({
            "eval_id": eval_count,
            "iteration": 0,
            "agent_idx": i,
            "position": pos.tolist(),
            "hyperparameters": hp.to_dict(),
            "fitness": float(fitness),
            "gbest_fitness": float(alpha_fitness),
            "info": info,
        })
        logger.info(
            "GWO Init [%d/%d] | val_AUPRC=%.5f | alpha=%.5f | lr=%.5f, h=%s, latent=%d",
            i + 1, n_wolves, fitness, alpha_fitness, hp.lr, hp.hidden_dims, hp.latent_dim
        )

    iteration_history.append({
        "iteration": 0,
        "best_fitness": float(alpha_fitness),
        "mean_fitness": float(np.mean(fitnesses)),
    })

    # Optimization loop
    for it in range(1, n_iterations + 1):
        a = 2.0 * (1.0 - it / n_iterations)
        iter_fitnesses = []

        for i in range(n_wolves):
            # Alpha hunting
            r1 = rng.uniform(0, 1, size=dim)
            r2 = rng.uniform(0, 1, size=dim)
            A1 = 2.0 * a * r1 - a
            C1 = 2.0 * r2
            D_alpha = np.abs(C1 * alpha_pos - positions[i])
            X1 = alpha_pos - A1 * D_alpha

            # Beta hunting
            r1 = rng.uniform(0, 1, size=dim)
            r2 = rng.uniform(0, 1, size=dim)
            A2 = 2.0 * a * r1 - a
            C2 = 2.0 * r2
            D_beta = np.abs(C2 * beta_pos - positions[i])
            X2 = beta_pos - A2 * D_beta

            # Delta hunting
            r1 = rng.uniform(0, 1, size=dim)
            r2 = rng.uniform(0, 1, size=dim)
            A3 = 2.0 * a * r1 - a
            C3 = 2.0 * r2
            D_delta = np.abs(C3 * delta_pos - positions[i])
            X3 = delta_pos - A3 * D_delta

            # Update position
            new_pos = np.clip((X1 + X2 + X3) / 3.0, 0.0, 1.0)
            positions[i] = new_pos

            # Evaluate candidate
            eval_count += 1
            hp = decode_vector(new_pos)
            fitness, info = eval_fn(hp)
            fitnesses[i] = fitness
            iter_fitnesses.append(fitness)

            # Update Alpha, Beta, Delta
            if fitness > alpha_fitness:
                delta_pos = beta_pos.copy()
                delta_fitness = beta_fitness
                beta_pos = alpha_pos.copy()
                beta_fitness = alpha_fitness
                alpha_pos = new_pos.copy()
                alpha_fitness = fitness
                alpha_hp = hp
            elif fitness > beta_fitness:
                delta_pos = beta_pos.copy()
                delta_fitness = beta_fitness
                beta_pos = new_pos.copy()
                beta_fitness = fitness
            elif fitness > delta_fitness:
                delta_pos = new_pos.copy()
                delta_fitness = fitness

            eval_history.append({
                "eval_id": eval_count,
                "iteration": it,
                "agent_idx": i,
                "position": new_pos.tolist(),
                "hyperparameters": hp.to_dict(),
                "fitness": float(fitness),
                "gbest_fitness": float(alpha_fitness),
                "info": info,
            })

            logger.info(
                "GWO Iter %d [%d/%d] | val_AUPRC=%.5f | alpha=%.5f",
                it, i + 1, n_wolves, fitness, alpha_fitness
            )

        iteration_history.append({
            "iteration": it,
            "best_fitness": float(alpha_fitness),
            "mean_fitness": float(np.mean(iter_fitnesses)),
        })

    elapsed_time = time.perf_counter() - start_time
    logger.info("GWO finished in %.2fs. Best val_AUPRC=%.5f", elapsed_time, alpha_fitness)

    return {
        "optimizer": "GWO",
        "best_position": alpha_pos.tolist(),
        "best_hyperparameters": alpha_hp.to_dict(),
        "best_fitness": float(alpha_fitness),
        "total_evaluations": eval_count,
        "n_wolves": n_wolves,
        "n_iterations": n_iterations,
        "elapsed_time_s": float(elapsed_time),
        "eval_history": eval_history,
        "iteration_history": iteration_history,
    }
