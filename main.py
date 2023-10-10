import pandas as pd

# Load the dataset into a pandas DataFrame
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
wine_data = pd.read_csv(url, sep=';')

# Display the first few rows of the dataset to understand its structure
print(wine_data.head())
# Check for missing values in the dataset
print(wine_data.isnull().sum())

# There are no missing values, so no imputation is needed.

# You might want to normalize the features if needed
# For example:
# wine_data = (wine_data - wine_data.min()) / (wine_data.max() - wine_data.min())
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of wine quality scores
plt.figure(figsize=(8, 6))
sns.countplot(x='quality', data=wine_data)
plt.title('Distribution of Wine Quality Scores')
plt.xlabel('Quality Score')
plt.ylabel('Count')
plt.show()

# Correlation matrix to explore relationships between features and wine quality
correlation_matrix = wine_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
# What is the distribution of the wine quality scores?
quality_counts = wine_data['quality'].value_counts()
print("Distribution of Wine Quality Scores:")
print(quality_counts)

# What is the relationship between different features and wine quality?
# Let's analyze the correlation between features and wine quality
correlations = wine_data.corr()['quality'].drop('quality')
print("\nCorrelation of Features with Wine Quality:")
print(correlations)

# What are the most important factors that influence the quality of wine?
# Let's consider features with significant absolute correlation values
important_features = correlations[abs(correlations) > 0.2]
print("\nImportant Factors Influencing Wine Quality:")
print(important_features)
