import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Load the dataset
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
zip_path = os.path.join(BASE_DIR, "zomato.zip")
data = pd.read_csv(zip_path, compression='zip')


data.rate = data.rate.replace("NEW", np.nan)
data.dropna(how='any', inplace=True)
del data['url']
del data['address']
del data['phone']
del data['location']
data.rename(columns={'approx_cost(for two people)': 'average_cost', 'listed_in(city)': 'locality', 'listed_in(type)': 'restaurant_type'}, inplace=True)
data.head()

X = data
X.rate = X.rate.astype(str)
X.rate = X.rate.apply(lambda x: x.replace('/5', ''))
X.rate = X.rate.apply(lambda x: float(x))
X.head()

rcParams['figure.figsize'] = 15, 7

# Countplot for locality
g = sns.countplot(x="locality", data=data, palette="Set1", hue=None)
g.set_xticklabels(g.get_xticklabels(), rotation=45, ha="right",fontsize=8)
plt.title('Locality', size=20)
plt.tight_layout()
plt.savefig("static/graphs/graph1.png")
plt.close()

# 1. Restaurant Chains with the Highest Number of Restaurants
restaurant_counts = data['name'].value_counts().head(10)
plt.figure(figsize=(16, 7))
sns.barplot(x=restaurant_counts.values, y=restaurant_counts.index, palette='viridis')
plt.title('Top 10 Restaurant Chains with the Highest Number of Restaurants')
plt.xlabel('Number of Restaurants')
plt.ylabel('Restaurant Chain')
plt.tight_layout()
plt.savefig("static/graphs/graph2.png")
plt.close()

# Table booking vs rate
x = pd.crosstab(X['rate'], X['book_table'])
x.div(x.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, color=['red', 'yellow'])
plt.title('Table Booking vs Rate', fontweight=30, fontsize=20)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("static/graphs/graph3.png")
plt.close()

# Distribution plot for average cost
X.average_cost = X.average_cost.apply(lambda x: x.replace(',', ''))
X.average_cost = X.average_cost.astype(int)
fig, ax = plt.subplots(figsize=[16, 4])
sns.histplot(X['average_cost'], kde=True, ax=ax)  
ticks = np.arange(0, X['average_cost'].max() + 1, 500) 
plt.xticks(ticks=ticks, rotation=45) 
ax.set_title('Cost Distribution for All Restaurants')
plt.tight_layout()
plt.savefig("static/graphs/graph4.png")
plt.close()

# Pie chart for restaurant types
restaurantTypeCount = data['restaurant_type'].value_counts().sort_values(ascending=True)
slices = restaurantTypeCount.values
labels = restaurantTypeCount.index
colors = ['#3333cc', '#ffff1a', '#ff3333', '#c2c2d6', '#6699ff', '#c4ff4d', '#339933']
plt.pie(slices, colors=colors, labels=labels, autopct='%1.0f%%', pctdistance=.5, labeldistance=1.2, shadow=True)
plt.title("Percentage of Restaurants According to Their Type", bbox={'facecolor': 'lightgray', 'pad': 5})
plt.gcf().set_size_inches(12, 12)
plt.tight_layout()
plt.savefig("static/graphs/graph5.png")
plt.close()

dish_liked_text = ' '.join(data['dish_liked'].dropna().values)
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='tab20').generate(dish_liked_text)
plt.figure(figsize=(16, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Most Liked Dishes')
plt.axis('off')
plt.tight_layout()
plt.savefig("static/graphs/graph6.png")
plt.close()