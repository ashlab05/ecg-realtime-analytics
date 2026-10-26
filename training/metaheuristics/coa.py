"""Coati Optimization Algorithm (COA) for Autoencoder Hyperparameter Tuning."""

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

logger = logging.getLogger("coa")


def run_coa(
    eval_fn: Callable[[Hyperparameters], Tuple[float, Dict]],
    n_coatis: int = 6,
    n_iterations: int = 5,
    seed: int = 42,
) -> Dict:
    """Run Coati Optimization Algorithm over the continuous [0, 1]^D search space.

    Enforces the exact same candidate evaluation budget as PSO and GWO:
    Initial evaluation: n_coatis
    Per iteration: n_coatis candidate evaluations
    Total evaluations = n_coatis * (n_iterations + 1) [or identical budgeted count].
    """
    rng = np.random.default_rng(seed)
    dim = DIMENSION

    # Initialize positions in [0, 1]
    positions = rng.uniform(0.05, 0.95, size=(n_coatis, dim))
    fitnesses = np.full(n_coatis, -float("inf"))

    eval_history: List[Dict] = []
    iteration_history: List[Dict] = []

    best_pos = None
    best_fitness = -float("inf")
    best_hp = None

    eval_count = 0
    start_time = time.perf_counter()

    logger.info("Initializing COA population (N=%d, T=%d, Dim=%d)...", n_coatis, n_iterations, dim)

    # Initial evaluation
    for i in range(n_coatis):
        eval_count += 1
        pos = positions[i]
        hp = decode_vector(pos)
        fitness, info = eval_fn(hp)
        fitnesses[i] = fitness

        if fitness > best_fitness:
            best_fitness = fitness
            best_pos = pos.copy()
            best_hp = hp

        eval_history.append({
            "eval_id": eval_count,
            "iteration": 0,
            "agent_idx": i,
            "position": pos.tolist(),
            "hyperparameters": hp.to_dict(),
            "fitness": float(fitness),
            "gbest_fitness": float(best_fitness),
            "info": info,
        })
        logger.info(
            "COA Init [%d/%d] | val_AUPRC=%.5f | gbest=%.5f | lr=%.5f, h=%s, latent=%d",
            i + 1, n_coatis, fitness, best_fitness, hp.lr, hp.hidden_dims, hp.latent_dim
        )

    iteration_history.append({
        "iteration": 0,
        "best_fitness": float(best_fitness),
        "mean_fitness": float(np.mean(fitnesses)),
    })

    # Optimization loop
    for it in range(1, n_iterations + 1):
        iter_fitnesses = []

        for i in range(n_coatis):
            # Phase 1: Exploration (Hunting iguanas) vs Phase 2: Exploitation (Evading predators)
            # Alternating or combined behavior based on coati role
            r = rng.uniform(0, 1, size=dim)
            I = rng.choice([1, 2])

            if i < n_coatis // 2:
                # Tree climbers chasing the best agent (iguana on the tree)
                new_pos = positions[i] + r * (best_pos - I * positions[i])
            else:
                # Ground coatis: local exploitation and evasion
                R = 0.2 * (1.0 - it / n_iterations)
                rand_step = (1.0 - 2.0 * rng.uniform(0, 1, size=dim)) * R
                new_pos = positions[i] + rand_step * positions[i]

            new_pos = np.clip(new_pos, 0.0, 1.0)

            # Evaluate candidate
            eval_count += 1
            hp = decode_vector(new_pos)
            fitness, info = eval_fn(hp)
            iter_fitnesses.append(fitness)

            # Greedy selection
            if fitness > fitnesses[i]:
                positions[i] = new_pos.copy()
                fitnesses[i] = fitness

            if fitness > best_fitness:
                best_fitness = fitness
                best_pos = new_pos.copy()
                best_hp = hp

            eval_history.append({
                "eval_id": eval_count,
                "iteration": it,
                "agent_idx": i,
                "position": new_pos.tolist(),
                "hyperparameters": hp.to_dict(),
                "fitness": float(fitness),
                "gbest_fitness": float(best_fitness),
                "info": info,
            })

            logger.info(
                "COA Iter %d [%d/%d] | val_AUPRC=%.5f | gbest=%.5f",
                it, i + 1, n_coatis, fitness, best_fitness
            )

        iteration_history.append({
            "iteration": it,
            "best_fitness": float(best_fitness),
            "mean_fitness": float(np.mean(fitnesses)),
        })

    elapsed_time = time.perf_counter() - start_time
    logger.info("COA finished in %.2fs. Best val_AUPRC=%.5f", elapsed_time, best_fitness)

    return {
        "optimizer": "COA",
        "best_position": best_pos.tolist(),
        "best_hyperparameters": best_hp.to_dict(),
        "best_fitness": float(best_fitness),
        "total_evaluations": eval_count,
        "n_coatis": n_coatis,
        "n_iterations": n_iterations,
        "elapsed_time_s": float(elapsed_time),
        "eval_history": eval_history,
        "iteration_history": iteration_history,
    }
