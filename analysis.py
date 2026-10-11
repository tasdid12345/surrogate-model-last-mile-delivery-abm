"""Surrogate modelling of a small last-mile delivery ABM: linear regression vs tuned LightGBM.
Target: van cost per parcel minus cargo-bike cost per parcel (EUR). Seed 42 everywhere."""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import qmc
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor

SEED, N_RUNS, SIDE = 42, 2000, 4.0  # city is a SIDE x SIDE km Manhattan grid
# uncertain assumptions: name -> (low, high)
BOUNDS = {"density": (5, 40), "van_speed": (12, 35), "bike_speed": (9, 20), "bike_cap": (8, 40),
          "fail_rate": (0.0, 0.25), "park_min": (0.5, 6.0), "service_min": (1.0, 5.0),
          "depot_km": (0.5, 8.0), "wage": (14, 30)}
VAN_CAP, SHIFT_H = 150, 8.0


def route_km(pts, depot):
    """Nearest-neighbour tour length (Manhattan) from the depot through pts and back."""
    left, cur, length = np.ones(len(pts), bool), depot, 0.0
    for _ in range(len(pts)):
        d = np.abs(pts - cur).sum(1)
        d[~left] = np.inf
        j = int(np.argmin(d)); length += d[j]; cur = pts[j]; left[j] = False
    return length + np.abs(cur - depot).sum()


def mode_cost(pts, att, depot, cap, speed, stop_min, load_min, wage, km_cost, veh_day):
    """Cost per parcel: sweep-clustered tours of <= cap parcels, labour + distance + vehicle-day costs."""
    ang = np.arctan2(pts[:, 1] - SIDE / 2, pts[:, 0] - SIDE / 2)
    o = np.argsort(ang); pts, att = pts[o], att[o]
    km, hours = 0.0, 0.0
    for s in range(0, len(pts), cap):
        k = route_km(pts[s:s + cap], depot)
        km += k
        hours += k / speed + (att[s:s + cap].sum() * stop_min + load_min) / 60
    vehicles = np.ceil(hours / SHIFT_H)
    return (wage * hours + km_cost * km + veh_day * vehicles) / len(pts)


def simulate(p, rng):
    """One ABM run: same customers/failures for both modes; returns (van, bike) cost per parcel."""
    n = int(p["density"] * SIDE ** 2)
    pts = rng.uniform(0, SIDE, (n, 2))
    att = 1 + (rng.random(n) < p["fail_rate"])  # a failed attempt wastes a stop
    depot = np.array([-p["depot_km"], SIDE / 2])
    van = mode_cost(pts, att, depot, VAN_CAP, p["van_speed"], p["park_min"] + p["service_min"],
                    15, p["wage"], 0.35, 70)
    bike = mode_cost(pts, att, depot, int(p["bike_cap"]), p["bike_speed"],
                     0.3 + 1.2 * p["service_min"], 8, p["wage"], 0.03, 10)
    return van, bike


def metrics(y, pred):
    return {"rmse": float(np.sqrt(mean_squared_error(y, pred))), "r2": float(r2_score(y, pred)),
            "acc": float(np.mean((pred > 0) == (y > 0)))}  # acc: which mode is cheaper


print("Running agent-based simulations...")
rng = np.random.default_rng(SEED)
u = qmc.LatinHypercube(d=len(BOUNDS), seed=SEED).random(N_RUNS)
lo, hi = np.array(list(BOUNDS.values())).T
X = pd.DataFrame(qmc.scale(u, lo, hi), columns=list(BOUNDS))
X["bike_cap"] = X["bike_cap"].round()
costs = np.array([simulate(r, rng) for r in X.to_dict("records")])
y = costs[:, 0] - costs[:, 1]  # van minus bike cost gap
print(f"{N_RUNS} runs done; gap mean {y.mean():.2f}, sd {y.std():.2f}, share bike cheaper {np.mean(y > 0):.2f}")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

# BASELINE: main-effects linear regression surrogate (deterministic)
lin = make_pipeline(StandardScaler(), LinearRegression()).fit(Xtr, ytr)
base = metrics(yte, lin.predict(Xte))
print("Baseline:", base)

# IMPROVED: LightGBM tuned by 3-fold CV on the training set only; 3 seeds
space = {"num_leaves": [4, 8, 16, 31], "learning_rate": [0.03, 0.05, 0.1], "n_estimators": [200, 400, 800],
         "min_child_samples": [5, 10, 20], "subsample": [0.7, 0.9], "colsample_bytree": [0.7, 1.0]}
imp_runs = []
for seed in (42, 43, 44):
    gb = LGBMRegressor(subsample_freq=1, random_state=seed, n_jobs=2, verbose=-1)
    cv = RandomizedSearchCV(gb, space, n_iter=15, cv=3, scoring="neg_root_mean_squared_error",
                            random_state=seed, n_jobs=1).fit(Xtr, ytr)
    imp_runs.append(metrics(yte, cv.best_estimator_.predict(Xte)))
    print(f"Improved seed {seed}:", imp_runs[-1], cv.best_params_)
imp = {k: float(np.mean([r[k] for r in imp_runs])) for k in base}
imp_sd = {k: float(np.std([r[k] for r in imp_runs])) for k in base}

if not all(np.isfinite(list(base.values()) + list(imp.values()))) or y.std() <= 0:
    sys.exit("Could not produce meaningful metrics")

os.makedirs("figures", exist_ok=True)
plt.bar(["Linear regression", "LightGBM"], [base["rmse"], imp["rmse"]], color=["gray", "tab:green"])
plt.ylabel("Test RMSE of cost gap (EUR/parcel)"); plt.tight_layout()
plt.savefig("figures/rmse.png", dpi=120)

res = {
    "question": "Can a gradient-boosting surrogate trained on runs of a simple last-mile delivery agent-based simulation predict the van-minus-bike cost gap under uncertain assumptions better than a linear regression surrogate?",
    "method": "Numpy ABM of a city grid (customers, parcels, couriers) comparing van and cargo-bike delivery; 9 uncertain assumptions sampled by Latin hypercube over 2000 runs; 80/20 split by run; models compared on test RMSE of the cost gap.",
    "dataset": {"name": "Self-generated last-mile delivery ABM runs", "source_url": "local numpy simulation (no download)",
                "n_rows": int(len(y)), "n_train": int(len(ytr)), "n_test": int(len(yte)),
                "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "Test RMSE of van-minus-bike cost gap (EUR per parcel)",
    "higher_is_better": False,
    "baseline": "Main-effects linear regression surrogate on the standardized assumptions.",
    "improved": "LightGBM surrogate with hyper-parameters chosen by 3-fold CV random search on the training runs (seeds 42-44).",
    "metrics": {"baseline_score": base["rmse"], "improved_score": imp["rmse"],
                "improvement": base["rmse"] - imp["rmse"],
                "baseline_score_std": 0.0, "improved_score_std": imp_sd["rmse"],
                "baseline_r2": base["r2"], "improved_r2": imp["r2"], "improved_r2_std": imp_sd["r2"],
                "baseline_cheaper_mode_accuracy": base["acc"], "improved_cheaper_mode_accuracy": imp["acc"],
                "improved_cheaper_mode_accuracy_std": imp_sd["acc"],
                "gap_std_all_runs": float(y.std()), "share_bike_cheaper": float(np.mean(y > 0))},
    "notes": "The ABM is a stylised toy (Manhattan grid, nearest-neighbour tours, sweep clustering, invented cost parameters), not the paper's model, so the results show the surrogate comparison only, not real delivery economics. Linear regression is deterministic (std 0); LightGBM mean/std over 3 seeds. Single train/test split; the test set is used only for the final evaluation."}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print("Wrote results.json")
