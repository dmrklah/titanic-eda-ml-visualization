# Basics of Machine Learning and Visualization — Titanic EDA

**Module:** Basics of Machine Learning and Visualization  
**Degree:** BSc Computer Science and Digitization

---

## Overview

Exploratory Data Analysis (EDA) on the Titanic dataset using Tableau for visualization and Python for reproducible analysis. The project investigates key factors influencing passenger survival, including age, gender, passenger class, and fare.

---

## Key Findings

| Factor | Finding |
|--------|---------|
| Passenger class | 1st class had the highest survival rate; 3rd class had the highest fatality rate |
| Gender | Female passengers survived at a much higher rate ("Women and Children First" policy) |
| Age | Majority of victims were under 30; older passengers in 1st/2nd class had better survival chances |
| Fare | 1st class paid significantly higher fares; fare strongly correlated with survival |

---

## Visualizations (9 Figures)

1. Age distribution of passengers (histogram)
2. Age by gender
3. Passenger class category
4. Passenger gender distribution
5. Survival status (pie chart) — 500 survived, 809 did not
6. Survival rates by passenger class
7. Survival rates by gender
8. Survival rates by age
9. Fare and class distribution (boxplot)

---

## Files

| File | Description |
|------|-------------|
| `titanic_eda.py` | Python EDA script — all 9 figures reproduced with Matplotlib/Seaborn |
| `report.pdf` | Full assignment report with Tableau visualizations and analysis |

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn
python titanic_eda.py
```

Dataset is loaded automatically via `seaborn.load_dataset("titanic")` — no manual download needed.

---

## Technologies

- Python · Pandas · Matplotlib · Seaborn
- Tableau (original visualizations in report)
