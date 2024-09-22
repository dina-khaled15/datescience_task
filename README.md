# datescience_task
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Read the data
data = pd.read_csv("data.csv")

# Data visualization
# Create a pairplot to explore the data
sns.pairplot(data, hue='Outcome')
plt.show()

# Scatter plot of SkinThickness vs Insulin
plt.scatter(df['SkinThickness'], df['Insulin'], marker='<')
plt.title('SkinThickness vs Insulin')  # Adding title
plt.xlabel('SkinThickness')  # Adding x-axis label
plt.ylabel('Insulin')  # Adding y-axis label
plt.show() 

# Model training
# Split the data and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)









  # The License:
   HESAHAM ASEM IS AREFERANCE FOR ME
   #THE YOUTUBE CHANNEL:https://youtube.com/playlist?list=PL6-3IRz2XF5UM-FWfQeF1_YhMMa12Eg3s&si=obEkLrgaSE-YM7N0
  # DOCUMANT:https://docs.python.org/3/
