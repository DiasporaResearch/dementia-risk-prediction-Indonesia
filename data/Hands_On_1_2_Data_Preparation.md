# Hands-On 1 & 2: Data Preparation — Copy-Paste Version

Copy each cell below into a new cell in your own Jupyter Notebook, in order.
Cells marked **(Markdown cell)** should be pasted into a Markdown cell (or just skipped — they are explanations, not code). Cells marked **(Code cell)** go into a Code cell.

---

## Cell 1 — (Markdown cell)

# Hands-On 1 & 2: Data Preparation
### ML-KC Study Workshop — Day 2

**Dataset:** `hypertension_dummy_dataset.csv` — 52,350 dummy hypertensive patients (plus a few intentional duplicates), built to look like the kind of dataset used in Dr Nuryunarsih's real hypertension prediction study.

**In this notebook we will:**
1. Check our setup
2. Load and inspect the dataset (shape, variable names, data types)
3. Check for missing values
4. Clean up variable (column) names
5. Look at descriptive statistics to spot outliers
6. Make some simple graphs
7. Recode Male/Female to 0/1
8. Create age categories
9. Handle missing values
10. Detect and handle outliers
11. Remove duplicate rows
12. Blend in the province mapping data (for regional analysis)
13. Save our cleaned dataset

> Work through the cells in order — each one builds on the last. Feel free to add your own cells to explore further!

---

## Cell 2 — (Markdown cell)

## Step 0: Setup Check

Before we touch any real data, let's make sure everyone's environment is working. Run the cell below.

If you're on **Google Colab**: this should just work.
If you're on your **own laptop with Jupyter**: this checks that the install video steps worked correctly.

---

## Cell 3 — (Code cell)

```python
# Setup check — run this first
import sys
print("Python version:", sys.version)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)
print("\nIf you can see version numbers above with no errors, you're ready to go!")
```

---

## Cell 4 — (Markdown cell)

**If you're using Google Colab and need to upload the CSV files:**
Uncomment and run the cell below, then choose `hypertension_dummy_dataset.csv` and `province_mapping.csv` from your computer when prompted.

---

## Cell 5 — (Code cell)

```python
# Uncomment these two lines if you're on Google Colab and need to upload files
# from google.colab import files
# uploaded = files.upload()
```

---

## Cell 6 — (Markdown cell)

## Step 1: Load the Dataset

Recall from yesterday: before we touch any data, we should be clear on our **outcome** (what are we trying to predict?) and how the data was **collected** (or in this case, simulated to resemble a real collection).

- **Outcome (y):** did the patient's systolic (SBP) and diastolic (DBP) blood pressure decrease after taking antihypertensive medication?
- **Predictors (X):** age, sex, job, height, weight, hypertension duration, smoking history, alcohol use, salty food consumption, exercise, diabetes, insomnia, sleeping pill use, and province.

Let's load it and take a first look.

---

## Cell 7 — (Code cell)

```python
df = pd.read_csv("hypertension_dummy_dataset.csv")
df.head()
```

---

## Cell 8 — (Markdown cell)

## Step 2: Check the Shape

How many rows (patients) and columns (variables) do we have?

---

## Cell 9 — (Code cell)

```python
print("Shape (rows, columns):", df.shape)
print(f"That's {df.shape[0]:,} patients and {df.shape[1]} variables.")
```

---

## Cell 10 — (Markdown cell)

## Step 3: Check Variable Names and Data Types

Notice anything messy about the column names? Extra spaces, inconsistent capitalisation... This is completely normal for real-world data, and it's the first thing we clean up.

---

## Cell 11 — (Code cell)

```python
print("Column names:")
for c in df.columns:
    print(f"  '{c}'")
```

---

## Cell 12 — (Code cell)

```python
df.dtypes
```

---

## Cell 13 — (Code cell)

```python
# .info() gives us a combined view: dtype + non-null counts in one place
df.info()
```

---

## Cell 14 — (Markdown cell)

**Notice:** the `Age` column shows up as an `object` (text) type, not a number. That's because a few rows contain things like `"52 yrs"` or `"unknown"` instead of a plain number — we'll need to fix that.

---

## Cell 15 — (Markdown cell)

## Step 4: Check for Missing Values

Let's see how much missing data we're dealing with, column by column.

---

## Cell 16 — (Code cell)

```python
missing_counts = df.isnull().sum()
missing_pct = (missing_counts / len(df) * 100).round(1)

missing_summary = pd.DataFrame({
    "missing_count": missing_counts,
    "missing_pct": missing_pct
}).sort_values("missing_count", ascending=False)

missing_summary[missing_summary["missing_count"] > 0]
```

---

## Cell 17 — (Markdown cell)

**Discussion point:** none of these columns are missing so much data that we'd need to drop them entirely, but we'll need a documented rule for handling missing values later on. Remember — always document *why* you're handling missing data a certain way.

---

## Cell 18 — (Markdown cell)

## Step 5: Clean Up Variable Names

Let's rename the columns to be short, consistent, and easy to type: all lowercase, no spaces, no special characters.

---

## Cell 19 — (Code cell)

```python
rename_map = {
    "Patient_ID": "patient_id",
    "Age ": "age",
    "sex ": "sex",
    "Job": "job",
    "height (cm)": "height_cm",
    "Weight_KG": "weight_kg",
    "HTN duration (yrs)": "htn_duration_years",
    "cigs_daily": "cigs_daily",
    "duration_smoking": "dur_smoking_years",
    "Type of Cigs": "type_cigarettes",
    "Alcohol ": "alcohol",
    "salty food": "salty_food",
    "Exercise": "exercise",
    "Diabetes": "diabetes",
    "Insomnia": "insomnia",
    "Sleeping_Pills": "sleeping_pills",
    "Province": "province",
    "SBP_before": "sbp_before",
    "DBP_before": "dbp_before",
    "SBP_after": "sbp_after",
    "DBP_after": "dbp_after",
    "SBP decreased?": "sbp_decreased",
    "DBP decreased?": "dbp_decreased",
}

df = df.rename(columns=rename_map)
df.columns.tolist()
```

---

## Cell 20 — (Markdown cell)

## Step 6: Fix the `age` Column

Before we can compute statistics on age, we need it to actually be numeric. Let's see what's going wrong.

---

## Cell 21 — (Code cell)

```python
# What non-numeric values are hiding in 'age'?
df["age"].apply(lambda x: isinstance(x, str)).sum()
```

---

## Cell 22 — (Code cell)

```python
# Show a few of the problem values
df.loc[df["age"].apply(lambda x: isinstance(x, str)), "age"].unique()
```

---

## Cell 23 — (Code cell)

```python
# Extract digits where possible, turn genuinely unusable text (like "unknown") into missing
df["age"] = (
    df["age"]
    .astype(str)
    .str.extract(r"(\d+)")[0]
)
df["age"] = pd.to_numeric(df["age"], errors="coerce")

print("age dtype is now:", df["age"].dtype)
print("Missing values in age after cleaning:", df["age"].isnull().sum())
```

---

## Cell 24 — (Markdown cell)

## Step 7: Descriptive Statistics — Spotting Outliers

`.describe()` is one of the fastest ways to sanity-check numeric data. Look closely at the **min** and **max** columns — do any of them look biologically implausible?

---

## Cell 25 — (Code cell)

```python
df.describe()
```

---

## Cell 26 — (Markdown cell)

**Questions to discuss with your neighbour:**
- Is a height of 999 cm plausible?
- Is a weight of 2 kg or 500 kg plausible for an adult?
- Is an SBP of 15 mmHg or 400 mmHg plausible?

These are exactly the kind of data-entry errors real datasets contain, and `.describe()` is often the first place you'll spot them.

---

## Cell 27 — (Markdown cell)

## Step 8: Simple Graphs to Visualise Outliers and Distributions

Numbers in a table can hide patterns that jump out immediately in a picture. Let's plot a few key variables.

---

## Cell 28 — (Code cell)

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df["age"].dropna(), bins=30, color="#1F4E79", edgecolor="white")
axes[0].set_title("Age distribution")
axes[0].set_xlabel("Age (years)")

axes[1].boxplot(df["height_cm"].dropna())
axes[1].set_title("Height (cm) — boxplot")
axes[1].set_ylabel("cm")

axes[2].boxplot(df["weight_kg"].dropna())
axes[2].set_title("Weight (kg) — boxplot")
axes[2].set_ylabel("kg")

plt.tight_layout()
plt.show()
```

---

## Cell 29 — (Markdown cell)

Notice the boxplots for height and weight — the little dots far away from the main box are exactly the kind of outliers `.describe()` warned us about.

---

## Cell 30 — (Markdown cell)

## Step 9: Standardise Text Categories

Before we recode anything, let's check what messy variants exist in our categorical columns.

---

## Cell 31 — (Code cell)

```python
print("Raw values in 'sex':")
print(df["sex"].value_counts(dropna=False))
```

---

## Cell 32 — (Code cell)

```python
# Standardise sex to just "Male" / "Female"
df["sex"] = df["sex"].str.strip().str.lower().map({
    "male": "Male", "m": "Male",
    "female": "Female", "f": "Female",
})
print(df["sex"].value_counts(dropna=False))
```

---

## Cell 33 — (Code cell)

```python
# Same idea for our Yes/No columns — several were entered inconsistently (Yes/yes/Y/1, No/no/N/0)
yn_columns = ["alcohol", "salty_food", "exercise", "diabetes", "insomnia", "sleeping_pills"]

yn_map = {
    "yes": "Yes", "y": "Yes", "1": "Yes",
    "no": "No", "n": "No", "0": "No",
}

for col in yn_columns:
    df[col] = df[col].astype(str).str.strip().str.lower().map(yn_map)
    # put back genuine missing values (astype(str) turns NaN into the string 'nan')
    df.loc[df[col].isna(), col] = np.nan

df[yn_columns].apply(lambda c: c.value_counts(dropna=False))
```

---

## Cell 34 — (Markdown cell)

## Step 10: Recode Male/Female to 0/1

Many models need numeric input, so let's create a numeric version of sex: `0 = Female`, `1 = Male`.

---

## Cell 35 — (Code cell)

```python
df["sex_male"] = df["sex"].map({"Female": 0, "Male": 1})
df[["sex", "sex_male"]].drop_duplicates().sort_values("sex")
```

---

## Cell 36 — (Markdown cell)

## Step 11: Create Age Categories

Continuous age is useful, but for some analyses (and for the association-rule-mining style work from yesterday) it helps to have age *groups* too.

---

## Cell 37 — (Code cell)

```python
bins = [0, 34, 44, 54, 64, 120]
labels = ["25-34 (early working age)", "35-44 (middle age)",
          "45-54 (pre-retirement age)", "55-64 (retirement age)", "65+ (elderly)"]

df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
df["age_group"].value_counts().sort_index()
```

---

## Cell 38 — (Markdown cell)

## Step 12: Handle Missing Values

Now that our columns are clean and consistent, let's decide what to do about missing values. There's no single right answer — the choice depends on the variable and how much is missing. A common, simple approach:

- **Numeric columns:** fill with the **median** (robust to outliers)
- **Categorical columns:** fill with the **mode** (most common category)

**Always document this choice** — it's exactly the kind of decision a data dictionary should record.

---

## Cell 39 — (Code cell)

```python
numeric_cols = ["age", "height_cm", "weight_kg", "cigs_daily", "dur_smoking_years"]
categorical_cols = ["job", "alcohol", "salty_food", "diabetes", "insomnia", "sleeping_pills", "province"]

for col in numeric_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"Filled '{col}' missing values with median = {median_val}")

for col in categorical_cols:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)
    print(f"Filled '{col}' missing values with mode = '{mode_val}'")

# age_group was created before we filled in missing 'age' values -- recompute it now
df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

print("\nRemaining missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
```

---

## Cell 40 — (Markdown cell)

## Step 13: Detect and Handle Outliers

Let's use the IQR (interquartile range) method — a common, simple rule of thumb — to flag outliers, and then cap the biologically implausible ones to a plausible clinical range.

---

## Cell 41 — (Code cell)

```python
def iqr_bounds(series, k=1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

for col in ["height_cm", "weight_kg", "sbp_before", "cigs_daily"]:
    low, high = iqr_bounds(df[col])
    n_outliers = ((df[col] < low) | (df[col] > high)).sum()
    print(f"{col}: plausible range ~[{low:.1f}, {high:.1f}] -> {n_outliers} flagged outliers")
```

---

## Cell 42 — (Code cell)

```python
# Cap clearly impossible values to plausible clinical ranges (rather than deleting the rows)
df["height_cm"] = df["height_cm"].clip(lower=130, upper=210)
df["weight_kg"] = df["weight_kg"].clip(lower=30, upper=180)
df["sbp_before"] = df["sbp_before"].clip(lower=90, upper=220)
df["cigs_daily"] = df["cigs_daily"].clip(lower=0, upper=60)

print("Capping complete. New ranges:")
df[["height_cm", "weight_kg", "sbp_before", "cigs_daily"]].describe().loc[["min", "max"]]
```

---

## Cell 43 — (Markdown cell)

## Step 14: Remove Duplicate Rows

A dataset built by combining medical records from several sites can easily end up with the same patient entered twice. Let's check.

---

## Cell 44 — (Code cell)

```python
n_dupes = df.duplicated().sum()
print(f"Found {n_dupes} fully duplicated rows.")

df = df.drop_duplicates().reset_index(drop=True)
print(f"Shape after removing duplicates: {df.shape}")
```

---

## Cell 45 — (Markdown cell)

**Note:** here we only checked for *fully identical* rows. In real projects, also check for duplicates on just `patient_id` — the same patient could be entered twice with slightly different values by mistake, which a full-row check would miss.

---

## Cell 46 — (Code cell)

```python
# Check for duplicate patient IDs specifically
dupe_ids = df["patient_id"].duplicated().sum()
print(f"Duplicate patient IDs (same ID appearing more than once): {dupe_ids}")
```

---

## Cell 47 — (Markdown cell)

## Step 15: Blend in the Province Mapping Data

So far, `province` gives us a lot of small categories (34 of them!) which can be hard to analyse directly. Let's bring in a lookup table that maps each province to a broader **region**, so we can support regional-level analysis too.

---

## Cell 48 — (Code cell)

```python
province_map = pd.read_csv("province_mapping.csv")
province_map.head()
```

---

## Cell 49 — (Code cell)

```python
df = df.merge(province_map, on="province", how="left")

print("Any provinces that failed to match?")
print(df[df["region"].isna()]["province"].unique())

df[["province", "region"]].drop_duplicates().sort_values("region").head(10)
```

---

## Cell 50 — (Code cell)

```python
df["region"].value_counts()
```

---

## Cell 51 — (Markdown cell)

## Step 16: Save the Cleaned Dataset

We're ready to save our clean, analysis-ready dataset. We'll use this file directly in Hands-On 3.

---

## Cell 52 — (Code cell)

```python
df.to_csv("hypertension_cleaned.csv", index=False)
print("Saved! Final shape:", df.shape)
df.head()
```

---

## Cell 53 — (Markdown cell)

## Guided Practice: Your Turn!

Now it's your turn to apply the same steps to your own dataset, with support from the facilitator. Work through this checklist:

1. Load your data with `pd.read_csv()` (or `pd.read_excel()` if it's an Excel file)
2. Check the shape — how many rows and columns?
3. Check the column names and data types — anything need renaming or converting?
4. Check for missing values — which columns are affected, and by how much?
5. Look at `.describe()` — do any min/max values look implausible?
6. Make one or two simple graphs (histogram or boxplot) of your key numeric variables
7. Standardise any messy categorical text (extra spaces, inconsistent case)
8. Recode any Yes/No or Male/Female-style variables to 0/1 if you plan to use them in a model
9. Create any category groupings you need (e.g. age groups)
10. Decide on — and document — your missing value strategy
11. Check for and remove duplicate rows
12. Save your cleaned file

**Raise your hand if you get stuck on any step — this is exactly the point of the exercise!**

---
