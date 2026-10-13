
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the CSV dataset
df = pd.read_csv("california_housing.csv")

# 2. Display the relationship between income and house value
plt.scatter(df["MedInc"], df["MedHouseVal"], alpha=0.2)

# 3. Add labels and a title
plt.xlabel("Median Income")
plt.ylabel("Median House Value ($100,000 units)")
plt.title("Income vs House Value in California")

# 4. Display the chart
plt.show()
