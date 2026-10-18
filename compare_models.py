
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

# 1. Load the dataset
df = pd.read_csv("california_housing.csv")

# 2. Separate features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# 3. Use the same split for a fair comparison
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Define the models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
}

# 5. Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print(f"MAE: {mae:.3f} (about ${mae * 100000:,.0f})")
    print(f"R²:  {r2:.3f}")
 # 6. Evaluate using 5-fold cross-validation
    cv_model = clone(model)

    cv_scores = cross_val_score(
        cv_model,
        X,
        y,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    print(f"5-Fold CV R²: {cv_scores.mean():.3f}")
    print(f"CV Std Dev:   {cv_scores.std():.3f}")
