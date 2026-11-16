# Water Potability ML

Can nine physico-chemical measurements (pH, hardness, sulfate, ...) predict whether a water sample is safe to drink?
This project answers that honestly: there is a weak, non-linear signal, and the dataset itself sets a low ceiling.

## Results (held-out test set, 656 samples, evaluated once)

Final model: Random Forest (tuned with 5-fold cross-validation), decision threshold chosen from out-of-fold
predictions so that precision for "potable" is at least 0.75.

| Metric | Value |
|---|---|
| ROC-AUC | 0.667 |
| PR-AUC (chance = 0.39) | 0.608 |
| Precision, "potable" | 0.80 |
| Recall, "potable" | 0.22 |
| Unsafe samples called potable | 14 of 400 (3.5%) |
| Accuracy | 0.68 |

Cross-validation on the development set (ROC-AUC, mean ± std over 5 folds):

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.486 ± 0.021 |
| Random Forest | 0.685 ± 0.016 (0.698 after tuning) |
| Gradient Boosting (HistGB) | 0.665 ± 0.016 |

## Why the threshold favours precision

The dangerous mistake is a **false positive**: declaring unsafe water potable. So the model is tuned to keep
precision for the "potable" class high, accepting that it confirms only about a fifth of the potable samples.
`TARGET_PRECISION` at the top of the notebook controls this trade-off.

## Method

1. Hold out 20% of the data as a test set (stratified), used once at the end.
2. Compare Logistic Regression, Random Forest and Gradient Boosting with stratified 5-fold CV.
   Median imputation (plus missing-value indicators) is part of each pipeline, so it is fitted inside each fold.
3. Tune the Random Forest with grid search on the same CV folds.
4. Choose the threshold from out-of-fold probabilities, never from data the model was trained on.
5. Refit on the full development set, evaluate on the test set, and compute permutation importance.

## Limitations

- The two classes have almost identical means on every feature and near-zero correlation with the label.
- Some values do not match real drinking-water guidance (e.g. total dissolved solids average ~22,000 ppm).
- Treat this as a teaching dataset. The model must not be used for real potability decisions.

## Data

`data/water_potability.csv`: 3,276 samples, 9 features, binary `Potability` label; missing values in
`ph` (15%), `Sulfate` (24%) and `Trihalomethanes` (5%). This is the public "Water Potability" dataset
distributed on Kaggle; check its page for the original source and licence before reusing it.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook water_potability_analysis.ipynb   # then Kernel > Restart & Run All
```

Or open the notebook in Google Colab and upload `data/water_potability.csv` into a `data/` folder.
Runtime is about one minute on a laptop; all randomness is fixed with `RANDOM_STATE = 42`.
Tested with Python 3.13, scikit-learn 1.9, pandas 3.1.
