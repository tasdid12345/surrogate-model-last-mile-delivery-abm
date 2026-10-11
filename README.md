# Surrogate models for a last-mile delivery agent-based simulation

## Motivation
The paper "Using an agent-based model to explore alternative modes of last-mile parcel delivery in urban contexts" (2018) uses an agent-based model (ABM) to compare delivery modes. ABM conclusions depend on many uncertain assumptions, and cheap surrogate models are often used to explore them. Here we test which kind of surrogate captures the simulation's behaviour better.

## Question
Can a gradient-boosting surrogate trained on runs of a simple last-mile delivery ABM predict the cost gap between delivery modes (van minus cargo bike) under uncertain assumptions better than a standard linear regression surrogate?

## Dataset
Self-generated: a small numpy ABM of a city grid with customers, parcels and couriers. Each of about 2,000 runs samples 9 uncertain assumptions (parcel density, van and bike speed, bike capacity, failed-delivery rate, parking search time, service time, depot distance, wage) and records the cost per parcel of each mode. This is not the paper's own data or model; no download is needed. Train/test split is 80/20 by run.

## Baseline vs improved
- Baseline: main-effects linear regression surrogate (the common ABM meta-model).
- Improved: LightGBM surrogate tuned by cross-validation on the training runs only, run with seeds 42, 43, 44.

Main metric: test RMSE of the cost gap (lower is better). Secondary: R² and accuracy of predicting which mode is cheaper.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> S Wise, K Cheliotis, O Bates, A Friday, J Allen (2018). *Using an agent-based model to explore alternative modes of last-mile parcel delivery in urban contexts*. Proceedings of the 1st. https://dl.acm.org/doi/abs/10.1145/3284038.3284039

Paper author: Prof. Adrian Friday, Lancaster University (a.friday@lancaster.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
