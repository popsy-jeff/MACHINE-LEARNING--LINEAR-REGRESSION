"""
Data Cleaning Pipeline - Titanic Dataset
(Same dataset as Kaggle's "Titanic - Machine Learning from Disaster")
=========================================================================
Install requirements first if needed:
    pip install pandas numpy scikit-learn seaborn imbalanced-learn --break-system-packages
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------
# 1. LOAD THE DATASET
# -------------------------------------------------------------------
import seaborn as sns
df = sns.load_dataset('titanic')   # same data as Kaggle's train.csv

# Or, if you download the Kaggle CSV directly:
# df = pd.read_csv('train.csv')

print(df.shape)
print(df.head())
print(df.info())
print(df.describe())

# -------------------------------------------------------------------
# 2. HANDLING MISSING VALUES
# -------------------------------------------------------------------
print(df.isnull().sum())                          # count missing per column
print(df.isnull().mean() * 100)                    # % missing per column

# Drop a column that's mostly missing (deck: 688/891 missing)
df = df.drop(columns=['deck'])

# Impute numeric column with median
df['age'] = df['age'].fillna(df['age'].median())

# Impute categorical column with mode
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])

# Drop rows still containing nulls (if any remain)
df = df.dropna()

# -------------------------------------------------------------------
# 3. REMOVING DUPLICATES
# -------------------------------------------------------------------
print(df.duplicated().sum())        # count duplicate rows
df = df.drop_duplicates()

# -------------------------------------------------------------------
# 4. FIXING DATA TYPES
# -------------------------------------------------------------------
print(df.dtypes)

df['pclass'] = df['pclass'].astype('category')
df['survived'] = df['survived'].astype('int')
df['sex'] = df['sex'].astype('category')

# Example: converting a date string column (if present)
# df['join_date'] = pd.to_datetime(df['join_date'], errors='coerce')

# -------------------------------------------------------------------
# 5. HANDLING OUTLIERS (using IQR method on 'fare')
# -------------------------------------------------------------------
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Outlier bounds for fare: {lower_bound:.2f} to {upper_bound:.2f}")

# Option A: Remove outliers
df_no_outliers = df[(df['fare'] >= lower_bound) & (df['fare'] <= upper_bound)]

# Option B: Cap outliers instead of removing (winsorizing)
df['fare'] = np.clip(df['fare'], lower_bound, upper_bound)

# -------------------------------------------------------------------
# 6. STANDARDIZING TEXT / CATEGORICAL DATA
# -------------------------------------------------------------------
df['sex'] = df['sex'].astype(str).str.lower().str.strip()
df['embark_town'] = df['embark_town'].astype(str).str.title().str.strip()

# -------------------------------------------------------------------
# 7. FIXING INCONSISTENT FORMATTING
# -------------------------------------------------------------------
# Example pattern for messy currency/number columns:
# df['salary'] = df['salary'].replace('[\$,]', '', regex=True).astype(float)

# Example pattern for messy date formats:
# df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)

# -------------------------------------------------------------------
# 8. FEATURE ENCODING & SCALING
# -------------------------------------------------------------------
# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=['sex', 'embarked', 'class', 'who'], drop_first=True)

# Label encode a binary/ordinal column
le = LabelEncoder()
df_encoded['alive'] = le.fit_transform(df['alive'])

# Scale numeric features
scaler = StandardScaler()
num_cols = ['age', 'fare', 'sibsp', 'parch']
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

# -------------------------------------------------------------------
# 9. HANDLING CLASS IMBALANCE (example, for classification target 'survived')
# -------------------------------------------------------------------
print(df_encoded['survived'].value_counts(normalize=True))

# Using SMOTE (requires: pip install imbalanced-learn --break-system-packages)
# from imblearn.over_sampling import SMOTE
# X = df_encoded.drop(columns=['survived'])
# y = df_encoded['survived']
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# -------------------------------------------------------------------
# 10. TRAIN / TEST SPLIT
# -------------------------------------------------------------------
X = df_encoded.drop(columns=['survived', 'alive'])
X = X.select_dtypes(include=[np.number])   # keep only numeric/encoded columns
y = df_encoded['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# -------------------------------------------------------------------
# 11. FINAL VALIDATION
# -------------------------------------------------------------------
print(df_encoded.isnull().sum().sum(), "missing values remaining")
print(df_encoded.duplicated().sum(), "duplicate rows remaining")
print(df_encoded.dtypes)

print("\nCleaned dataset ready for analysis/ML.")
