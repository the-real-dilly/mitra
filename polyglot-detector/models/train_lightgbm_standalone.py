import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# feature_rows: list of dicts from extract_features(), one per file
# labels: list of 0/1, 1 = polyglot
df = pd.DataFrame(feature_rows)
df["label"] = labels

X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "verbose": -1,
}

model = lgb.train(
    params,
    train_set,
    num_boost_round=500,
    valid_sets=[val_set],
    callbacks=[lgb.early_stopping(stopping_rounds=30)],
)

preds = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, preds))

# Feature importance — useful for auditing what the model keys on
importance = pd.Series(model.feature_importance(), index=X.columns)
print(importance.sort_values(ascending=False))

model.save_model("polyglot_detector.txt")