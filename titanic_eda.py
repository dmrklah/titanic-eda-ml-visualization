"""
Basics of Machine Learning and Visualization
Module: Basics of Machine Learning and Visualization
Degree: BSc Computer Science and Digitization

Description:
  Exploratory Data Analysis (EDA) on the Titanic dataset.
  Reproduces the key visualizations from the assignment using Python/Matplotlib
  as a complement to the Tableau analysis.

Dataset: Titanic (available via seaborn or Kaggle)

Requirements: pip install pandas numpy matplotlib seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load Dataset ─────────────────────────────────────────────

def load_data():
    """Load Titanic dataset via seaborn (mirrors Kaggle structure)."""
    df = sns.load_dataset("titanic")

    # Rename columns to match assignment variable names
    df = df.rename(columns={
        "survived": "Survived",
        "pclass":   "Pclass",
        "sex":      "Sex",
        "age":      "Age",
        "fare":     "Fare",
        "sibsp":    "SibSp",
        "parch":    "Parch",
        "embark_town": "Embarked",
    })

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
    return df


# ── Data Cleaning ─────────────────────────────────────────────

def clean_data(df):
    """Handle missing values and prepare data."""
    df = df.copy()

    # Fill missing Age with median
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Fill missing Fare with median
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Drop rows with missing Embarked
    df = df.dropna(subset=["Embarked"])

    print(f"\nAfter cleaning: {df.shape[0]} rows")
    return df


# ── Figure 1: Age Distribution (Histogram) ───────────────────

def plot_age_distribution(df):
    plt.figure(figsize=(9, 5))
    plt.hist(df["Age"].dropna(), bins=20, color="#2E86AB", edgecolor="white", alpha=0.85)
    plt.title("Figure 1: Age of Passengers")
    plt.xlabel("Age")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()
    plt.savefig("fig1_age_distribution.png", dpi=150)
    plt.show()
    print("Saved: fig1_age_distribution.png")


# ── Figure 2: Age by Gender ───────────────────────────────────

def plot_age_by_gender(df):
    plt.figure(figsize=(9, 5))
    bins = range(0, 90, 10)
    for gender, color in [("male", "#2E86AB"), ("female", "#E84855")]:
        subset = df[df["Sex"] == gender]["Age"].dropna()
        plt.hist(subset, bins=bins, alpha=0.6, label=gender.capitalize(), color=color, edgecolor="white")
    plt.title("Figure 2: Age of Passengers by Gender")
    plt.xlabel("Age")
    plt.ylabel("Number of Passengers")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig2_age_by_gender.png", dpi=150)
    plt.show()
    print("Saved: fig2_age_by_gender.png")


# ── Figure 3: Passenger Class ─────────────────────────────────

def plot_passenger_class(df):
    class_counts = df["Pclass"].value_counts().sort_index()
    labels = ["1st Class", "2nd Class", "3rd Class"]
    colors = ["#F4A261", "#2E86AB", "#E84855"]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, class_counts.values, color=colors, edgecolor="white", alpha=0.85)
    plt.title("Figure 3: Passenger Class Category")
    plt.xlabel("Class")
    plt.ylabel("Number of Passengers")
    for i, v in enumerate(class_counts.values):
        plt.text(i, v + 5, str(v), ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig("fig3_passenger_class.png", dpi=150)
    plt.show()
    print("Saved: fig3_passenger_class.png")


# ── Figure 4: Passenger Gender ────────────────────────────────

def plot_passenger_gender(df):
    gender_counts = df["Sex"].value_counts()

    plt.figure(figsize=(6, 5))
    plt.bar(["Male", "Female"], [gender_counts.get("male", 0), gender_counts.get("female", 0)],
            color=["#2E86AB", "#E84855"], edgecolor="white", alpha=0.85)
    plt.title("Figure 4: Gender of Passengers")
    plt.xlabel("Gender")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()
    plt.savefig("fig4_passenger_gender.png", dpi=150)
    plt.show()
    print("Saved: fig4_passenger_gender.png")


# ── Figure 5: Survival Status (Pie Chart) ────────────────────

def plot_survival_pie(df):
    survival_counts = df["Survived"].value_counts()
    labels = ["Did Not Survive", "Survived"]
    colors = ["#E84855", "#2E86AB"]

    plt.figure(figsize=(6, 6))
    plt.pie(
        [survival_counts.get(0, 0), survival_counts.get(1, 0)],
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    plt.title("Figure 5: Survival Status of Passengers")
    plt.tight_layout()
    plt.savefig("fig5_survival_pie.png", dpi=150)
    plt.show()
    print("Saved: fig5_survival_pie.png")


# ── Figure 6: Survival by Passenger Class ────────────────────

def plot_survival_by_class(df):
    survival_class = df.groupby(["Pclass", "Survived"]).size().unstack()
    survival_class.index = ["1st Class", "2nd Class", "3rd Class"]
    survival_class.columns = ["Did Not Survive", "Survived"]

    survival_class.plot(kind="bar", color=["#E84855", "#2E86AB"],
                        edgecolor="white", alpha=0.85, figsize=(8, 5))
    plt.title("Figure 6: Survival Rates Across Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Number of Passengers")
    plt.xticks(rotation=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig6_survival_by_class.png", dpi=150)
    plt.show()
    print("Saved: fig6_survival_by_class.png")


# ── Figure 7: Survival by Gender ─────────────────────────────

def plot_survival_by_gender(df):
    survival_gender = df.groupby(["Sex", "Survived"]).size().unstack()
    survival_gender.index = ["Female", "Male"]
    survival_gender.columns = ["Did Not Survive", "Survived"]

    survival_gender.plot(kind="bar", color=["#E84855", "#2E86AB"],
                         edgecolor="white", alpha=0.85, figsize=(7, 5))
    plt.title("Figure 7: Survival Rates Across Genders")
    plt.xlabel("Gender")
    plt.ylabel("Number of Passengers")
    plt.xticks(rotation=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig7_survival_by_gender.png", dpi=150)
    plt.show()
    print("Saved: fig7_survival_by_gender.png")


# ── Figure 8: Survival by Age ─────────────────────────────────

def plot_survival_by_age(df):
    plt.figure(figsize=(9, 5))
    for survived, color, label in [(0, "#E84855", "Did Not Survive"), (1, "#2E86AB", "Survived")]:
        subset = df[df["Survived"] == survived]["Age"].dropna()
        plt.hist(subset, bins=20, alpha=0.6, color=color, label=label, edgecolor="white")
    plt.title("Figure 8: Survival Rates Across Ages")
    plt.xlabel("Age")
    plt.ylabel("Number of Passengers")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig8_survival_by_age.png", dpi=150)
    plt.show()
    print("Saved: fig8_survival_by_age.png")


# ── Figure 9: Fare and Class Distribution ────────────────────

def plot_fare_by_class(df):
    plt.figure(figsize=(8, 5))
    class_labels = {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}
    colors = ["#F4A261", "#2E86AB", "#E84855"]

    data = [df[df["Pclass"] == cls]["Fare"].dropna() for cls in [1, 2, 3]]
    bp = plt.boxplot(data, labels=["1st Class", "2nd Class", "3rd Class"],
                     patch_artist=True, medianprops={"color": "black"})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    plt.title("Figure 9: Fare and Class Distribution")
    plt.xlabel("Passenger Class")
    plt.ylabel("Fare (£)")
    plt.tight_layout()
    plt.savefig("fig9_fare_by_class.png", dpi=150)
    plt.show()
    print("Saved: fig9_fare_by_class.png")


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)

    plot_age_distribution(df)
    plot_age_by_gender(df)
    plot_passenger_class(df)
    plot_passenger_gender(df)
    plot_survival_pie(df)
    plot_survival_by_class(df)
    plot_survival_by_gender(df)
    plot_survival_by_age(df)
    plot_fare_by_class(df)

    print("\nAll figures saved. EDA complete.")
