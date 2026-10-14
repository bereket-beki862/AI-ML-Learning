import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = {
    "Math": [85, 90, 78, 92, 88],
    "Physics": [80, 85, 75, 89, 91],
    "Chemistry": [88, 92, 80, 85, 90],
    "Programming": [95, 98, 85, 90, 96]
}

df = pd.DataFrame(data)

correlation = df.corr()

sns.heatmap(correlation, annot=True, cmap="coolwarm")

plt.title("Subject Marks Correlation")

plt.show()