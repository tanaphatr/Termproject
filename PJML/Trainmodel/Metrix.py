import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the CSV file
file_path = r'E:\Term project\cleaned_data.csv'
data = pd.read_csv(file_path)

# Convert the 'sale_date' column to datetime if it's not already
data['sale_date'] = pd.to_datetime(data['sale_date'])

# Extract features from the date
data['day_of_year'] = data['sale_date'].dt.dayofyear
data['month'] = data['sale_date'].dt.month
data['day_of_week'] = data['sale_date'].dt.dayofweek

# Select only numeric columns for correlation
numeric_data = data.select_dtypes(include=['number'])

# Calculate the correlation matrix
correlation_matrix = numeric_data.corr()

# Save the correlation matrix to a CSV file
correlation_matrix.to_csv(r'E:\Term project\correlation_matrix.csv')

# Set the size of the plot
plt.figure(figsize=(10, 8))

# Create a heatmap of the correlation matrix
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})

# Add title
plt.title('Correlation Matrix')

# Save the figure
plt.savefig(r'E:\Term project\correlation_matrix.png', bbox_inches='tight')

# Show the plot
plt.show()
