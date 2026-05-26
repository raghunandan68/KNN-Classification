import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="KNN Classifier", layout="wide")

st.title("KNN Classification")

st.header("1. Data Ingestion")
st.success("✅ Wine Dataset loaded Successfully...")
@st.cache_data
def load_data():
    data = load_wine(as_frame=True)
    df = data.frame
    np.random.seed(42)
    for col in df.columns[:-1]:
        df.loc[df.sample(frac=0.1).index, col] = np.nan
    return df

df = load_data()
st.dataframe(df, use_container_width=True)

st.header("2. Data Cleaning")

strategy = st.selectbox("Missing Value Strategy", ["Mean", "Median", "Most Frequent", "Drop Rows"])

df_clean = df.copy()

if strategy == "Drop Rows":
    df_clean = df_clean.dropna()
else:
    fill_map = {
        "Mean": "mean",
        "Median": "median",
        "Most Frequent": "most_frequent"
    }

    imputer = SimpleImputer(strategy=fill_map[strategy])
    cols = df_clean.select_dtypes(include=np.number).columns
    df_clean[cols] = imputer.fit_transform(df_clean[cols])

st.dataframe(df_clean, use_container_width=True)

if st.button("Save Cleaned Dataset"):
    path = os.path.join(CLEAN_DIR, "cleaned_wine.csv")
    df_clean.to_csv(path, index=False)
    st.success("Dataset saved successfully")

st.header("3. Load Cleaned Dataset")

files = [f for f in os.listdir(CLEAN_DIR) if "wine" in f]

if not files:
    st.stop()

file = st.selectbox("Select Dataset", files)
data = pd.read_csv(os.path.join(CLEAN_DIR, file))

st.dataframe(data, use_container_width=True)

st.header("4. Model Training")

X = data.drop(columns=["target"])
y = data["target"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

model = grid.best_estimator_
pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

st.success(f"Accuracy: {acc:.2f}")

st.write("Best Parameters:", grid.best_params_)

st.write("Best CV Score:", grid.best_score_)

st.header("5. Confusion Matrix (Table)")

cm = confusion_matrix(y_test, pred)

cm_df = pd.DataFrame(
    cm,
    columns=[f"Pred {i}" for i in range(cm.shape[1])],
    index=[f"Actual {i}" for i in range(cm.shape[0])]
)

st.dataframe(cm_df, use_container_width=True)

st.header("6. Classification Report (Table)")

report = classification_report(y_test, pred, output_dict=True)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df, use_container_width=True)

st.header("7. Visualization")

fig, ax = plt.subplots()

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

ax.set_title("Confusion Matrix Heatmap")

st.pyplot(fig)

st.header("8. Save Model")

model_path = os.path.join(MODEL_DIR, "trained_models.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

st.success(f"Model saved at {model_path}")

st.header("9. Sample Predictions")

sample = pd.DataFrame({
    "Actual": y_test[:10],
    "Predicted": pred[:10]
})

st.dataframe(sample, use_container_width=True)