import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = {
    "Math": [85, 90, 78, 92, 88, 75],
    "Physics": [80, 85, 75, 89, 91, 70],
    "Programming": [95, 98, 85, 90, 96, 80]
}

df = pd.DataFrame(data)

sns.pairplot(df)

plt.show()