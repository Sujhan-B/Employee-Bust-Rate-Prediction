import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from google.colab import drive
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score

# Suppress warnings
warnings.filterwarnings('ignore')

# Mount Google Drive
drive.mount('/content/drive')

# Display settings
pd.set_option("display.max_columns", None)

# Load data
df = pd.read_csv("/content/drive/MyDrive/employee_burnout_analysis-AI 2.csv")
df["Date of Joining"] = pd.to_datetime(df["Date of Joining"])

# Overview
print(df.shape)
print(df.info())
print(df.isna().sum())

# Unique values and value counts
for col in df.columns:
    print(f'\nUnique values in {col}:\n{df[col].unique()}')
    print(f'\nValue counts of {col}:\n{df[col].value_counts()}\n')

# Drop unnecessary column
df.drop(['Employee ID'], axis=1, inplace=True)

# Skewness check
intfloatdf = df.select_dtypes([np.int64, np.float64])
for col in intfloatdf.columns:
    skew = intfloatdf[col].skew()
    if skew >= 0.1:
        print(f'\n{col} is positively skewed with value: {skew}')
    elif skew <= -0.1:
        print(f'\n{col} is negatively skewed with value: {skew}')
    else:
        print(f'\n{col} is normally distributed with skew value: {skew}')

# Handle missing values
df['Resource Allocation'].fillna(df['Resource Allocation'].mean(), inplace=True)
df['Burn Rate'].fillna(df['Burn Rate'].mean(), inplace=True)
df['Mental Fatigue Score'].fillna(df['Mental Fatigue Score'].mean(), inplace=True)

print(df.isna().sum())

# Correlation matrix
df_numeric = df.select_dtypes(include=['number'])
corr = df_numeric.corr()
sns.set(rc={'figure.figsize': (14, 12)})
fig = px.imshow(corr, text_auto=True, aspect='auto')
fig.show()

# Plot distributions
plt.figure(figsize=(10, 8))
sns.countplot(x='Gender', data=df, palette="magma")
plt.title('Distribution of Gender')
plt.show()

plt.figure(figsize=(10, 8))
sns.countplot(x='WFH Setup Available', data=df, palette="dark:salmon_r")
plt.title('Distribution of WFH Setup Available')
plt.show()

burn_st = df.loc[:, 'Date of Joining':'Burn Rate'].select_dtypes([int, float])
for col in burn_st.columns:
    fig = px.histogram(burn_st, x=col, title=f"Distribution of {col}", color_discrete_sequence=['indianred'])
    fig.update_layout(bargap=0.2)
    fig.show()

# Burn Rate by Designation
fig = px.line(df, y="Burn Rate", color="Designation", title="Burn Rate by Designation",
              color_discrete_sequence=px.colors.qualitative.Pastel1)
fig.update_layout(bargap=0.1)
fig.show()

# Burn Rate by Gender
fig = px.line(df, y="Burn Rate", color="Gender", title="Burn Rate by Gender",
              color_discrete_sequence=px.colors.qualitative.Pastel1)
fig.update_layout(bargap=0.1)
fig.show()

# Mental Fatigue Score by Designation
fig = px.line(df, y="Mental Fatigue Score", color="Designation", title="Mental Fatigue Score vs Designation",
              color_discrete_sequence=px.colors.qualitative.Pastel1)
fig.update_layout(bargap=0.1)
fig.show()

# Relational plot
sns.relplot(
    data=df,
    x="Designation",
    y="Mental Fatigue Score",
    col="Company Type",
    hue="Company Type",
    size="Burn Rate",
    style="Gender",
    palette=["g", "r"],
    sizes=(50, 200)
)
plt.show()

# Label encoding
label_encode = preprocessing.LabelEncoder()
df['Company_TypeLabel'] = label_encode.fit_transform(df['Company Type'])
df['WFH_Setup_AvailableLabel'] = label_encode.fit_transform(df['WFH Setup Available'])
df['GenderLabel'] = label_encode.fit_transform(df['Gender'])

# Final features
columns = ['Designation', 'Resource Allocation', 'Mental Fatigue Score',
           'GenderLabel', 'Company_TypeLabel', 'WFH_Setup_AvailableLabel']
X = df[columns]
y = df['Burn Rate']

# PCA
pca = PCA(0.95)
x_pca = pca.fit_transform(X)

print("PCA shape of X:", x_pca.shape, "Original shape:", X.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Number of components selected:", pca.n_components_)

# Train-test split
X_train_pca, X_test, Y_train, Y_test = train_test_split(x_pca, y, test_size=0.25, random_state=10)

# Random Forest Regressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train_pca, Y_train)

train_pred_rf = rf_model.predict(X_train_pca)
test_pred_rf = rf_model.predict(X_test)

print("Train Accuracy (RF):", round(100 * r2_score(Y_train, train_pred_rf), 2), "%")
print("Test Accuracy (RF):", round(100 * r2_score(Y_test, test_pred_rf), 2), "%")

# AdaBoost Regressor
abr_model = AdaBoostRegressor()
abr_model.fit(X_train_pca, Y_train)

train_pred_adaboost = abr_model.predict(X_train_pca)
test_pred_adaboost = abr_model.predict(X_test)

print("Train Accuracy (AdaBoost):", round(100 * r2_score(Y_train, train_pred_adaboost), 2), "%")
print("Test Accuracy (AdaBoost):", round(100 * r2_score(Y_test, test_pred_adaboost), 2), "%")
