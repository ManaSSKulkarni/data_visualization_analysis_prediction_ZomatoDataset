import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.title("Dataset(by Zomato) Analysis & ML Prediction")

# LOAD DATA
data = pd.read_csv("sampledataset.csv").sample(2000, random_state=42)

# CLEAN RATE COLUMN
data['rate'] = data['rate'].astype(str)

data['rate'] = (
    data['rate']
    .str.replace('/5', '', regex=False)
    .str.strip()
)

# Replace invalid values
data['rate'] = data['rate'].replace(['NEW', '-', 'nan'], np.nan)

# Convert to float
data['rate'] = data['rate'].astype(float)

# RENAME COLUMNS
data.rename(columns={
    'approx_cost(for two people)': 'average_cost',
    'listed_in(city)': 'locality',
    'listed_in(type)': 'restaurant_type'
}, inplace=True)

# CLEAN AVERAGE COST
data['average_cost'] = (
    data['average_cost']
    .astype(str)
    .str.replace(',', '', regex=False)
)

data['average_cost'] = pd.to_numeric(
    data['average_cost'],
    errors='coerce'
)

# REMOVE NULLS
data.dropna(how='any', inplace=True)

# DELETE UNUSED COLUMNS
del data['url']
del data['address']
del data['phone']
del data['location']

st.header("Dataset")

if st.checkbox("Show Dataset"):
    st.dataframe(data)

# -----------------------------
# DATA ANALYSIS SECTION
# -----------------------------

st.header("Data Analysis")

# GRAPH 1
st.subheader("Top 10 Localities")

top_localities = data['locality'].value_counts().head(10)

fig1, ax1 = plt.subplots(figsize=(12,6))

sns.barplot(
    x=top_localities.index,
    y=top_localities.values,
    ax=ax1
)

plt.xticks(rotation=45)

st.pyplot(fig1)

# GRAPH 2
st.subheader("Top Restaurant Chains")

restaurant_counts = data['name'].value_counts().head(10)

fig2, ax2 = plt.subplots(figsize=(12,6))

sns.barplot(
    x=restaurant_counts.values,
    y=restaurant_counts.index,
    ax=ax2
)

st.pyplot(fig2)

# GRAPH 3
st.subheader("Table Booking vs Rating")

X = data.copy()


x = pd.crosstab(X['rate'], X['book_table'])

fig3, ax3 = plt.subplots(figsize=(10,6))

x.div(x.sum(1).astype(float), axis=0).plot(
    kind='bar',
    stacked=True,
    ax=ax3
)

st.pyplot(fig3)

# GRAPH 4
st.subheader("Average Cost Distribution")


fig4, ax4 = plt.subplots(figsize=(12,5))

sns.histplot(X['average_cost'], kde=True, ax=ax4)

st.pyplot(fig4)

# GRAPH 5 WORDCLOUD
st.subheader("Most Liked Dishes")

dish_liked_text = ' '.join(data['dish_liked'].dropna().values)

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate(dish_liked_text)

fig5, ax5 = plt.subplots(figsize=(15,7))

ax5.imshow(wordcloud, interpolation='bilinear')
ax5.axis("off")

st.pyplot(fig5)

# -----------------------------
# MACHINE LEARNING SECTION
# -----------------------------

st.header("Machine Learning")

# ENCODING
label_encoder_name = LabelEncoder()
label_encoder_online_order = LabelEncoder()
label_encoder_book_table = LabelEncoder()
label_encoder_locality = LabelEncoder()
label_encoder_restaurant_type = LabelEncoder()

data['name'] = label_encoder_name.fit_transform(data['name'])
data['online_order'] = label_encoder_online_order.fit_transform(data['online_order'])
data['book_table'] = label_encoder_book_table.fit_transform(data['book_table'])
data['locality'] = label_encoder_locality.fit_transform(data['locality'])
data['restaurant_type'] = label_encoder_restaurant_type.fit_transform(data['restaurant_type'])

data['average_cost'] = (
    data['average_cost']
    .astype(str)
    .str.replace(',', '')
    .astype(float)
)

features = [
    'name',
    'online_order',
    'book_table',
    'average_cost',
    'locality',
    'restaurant_type'
]

X = data[features]
y = data['rate']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# MODELS
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    random_state=42
)

gb_model.fit(X_train, y_train)

# PREDICTIONS
lr_predictions = lr_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)

# RESULTS
results = {
    "Linear Regression": round(r2_score(y_test, lr_predictions), 4),
    "Random Forest": round(r2_score(y_test, rf_predictions), 4),
    "Gradient Boosting": round(r2_score(y_test, gb_predictions), 4)
}

st.subheader("Model Performance (R2 Score)")

st.write(results)

# GRAPH 6 ML COMPARISON
fig6, ax6 = plt.subplots(figsize=(10,6))

ax6.bar(results.keys(), results.values())

ax6.set_ylabel("R2 Score")

st.pyplot(fig6)

# GRAPH 7 ACTUAL VS PREDICTED
fig7, ax7 = plt.subplots(figsize=(10,6))

ax7.scatter(y_test, rf_predictions)

ax7.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)]
)

ax7.set_xlabel("Actual")
ax7.set_ylabel("Predicted")

st.pyplot(fig7)

# PREDICTION
st.subheader("Sample Restaurant Prediction")

# ORIGINAL VALUES
restaurant_details = {
    "Restaurant Name": "Kulkarni Restaurant",
    "Online Order": "Yes",
    "Table Booking": "Yes",
    "Average Cost for Two": 300,
    "Locality": "Banashankari",
    "Restaurant Type": "Bar"
}

# DISPLAY ORIGINAL VALUES
st.write("### New Restaurant Details")
st.write(restaurant_details)

# ENCODE VALUES FOR MODEL
new_restaurant = pd.DataFrame({
    'name': [
        label_encoder_name.transform(
            [restaurant_details["Restaurant Name"]]
        )[0]
        if restaurant_details["Restaurant Name"] in label_encoder_name.classes_
        else 0
    ],

    'online_order': [
        label_encoder_online_order.transform(
            [restaurant_details["Online Order"]]
        )[0]
    ],

    'book_table': [
        label_encoder_book_table.transform(
            [restaurant_details["Table Booking"]]
        )[0]
    ],

    'average_cost': [
        restaurant_details["Average Cost for Two"]
    ],

    'locality': [
        label_encoder_locality.transform(
            [restaurant_details["Locality"]]
        )[0]
        if restaurant_details["Locality"] in label_encoder_locality.classes_
        else 0
    ],

    'restaurant_type': [
        label_encoder_restaurant_type.transform(
            [restaurant_details["Restaurant Type"]]
        )[0]
        if restaurant_details["Restaurant Type"] in label_encoder_restaurant_type.classes_
        else 0
    ]
})

# PREDICTION
predicted_rating = rf_model.predict(new_restaurant)

st.success(
    f"Predicted User Rating: {round(predicted_rating[0], 2)}"
)