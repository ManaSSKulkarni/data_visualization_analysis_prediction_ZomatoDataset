import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Loading Dataset
data = pd.read_csv('./zomato.csv')
data.rate = data.rate.replace("NEW", np.nan)
data.dropna(how='any', inplace=True)
del data['url']
del data['address']
del data['phone']
del data['location']
data.rename(columns={'approx_cost(for two people)': 'average_cost', 'listed_in(city)': 'locality', 'listed_in(type)': 'restaurant_type'}, inplace=True)

# Data preprocessing
data['average_cost'] = data['average_cost'].str.replace(',', '').astype(float)
data.rate = data.rate.astype(str).apply(lambda x: x.replace('/5', '')).astype(float)

# Encoding 
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

# Feature selection
features = ['name', 'online_order', 'book_table', 'average_cost', 'locality', 'restaurant_type']
X = data[features]
y = data['rate']

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
print("Linear Regression")
print("MSE:", mean_squared_error(y_test, lr_predictions))
print("R2 Score:", r2_score(y_test, lr_predictions))

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print("\nRandom Forest")
print("MSE:", mean_squared_error(y_test, rf_predictions))
print("R2 Score:", r2_score(y_test, rf_predictions))

# Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
print("\nGradient Boosting")
print("MSE:", mean_squared_error(y_test, gb_predictions))
print("R2 Score:", r2_score(y_test, gb_predictions))

# Custom function 
def handle_new_label(encoder, label):
    if label not in encoder.classes_:
        return -1  
    
    return encoder.transform([label])[0]

# Best Model
best_model = rf_model

# New Restaurant
new_restaurant = pd.DataFrame({
    'name': [handle_new_label(label_encoder_name, 'Kulkarni RestoBar')],
    'online_order': [handle_new_label(label_encoder_online_order, 'Yes')],
    'book_table': [handle_new_label(label_encoder_book_table, 'Yes')],
    'average_cost': [300],
    'locality': [handle_new_label(label_encoder_locality, 'Banashankari')],
    'restaurant_type': [handle_new_label(label_encoder_restaurant_type, 'Bar')]
})

predicted_rating = best_model.predict(new_restaurant)
print("Predicted Rating for the new restaurant:", predicted_rating)

# Actual vs Predicted Ratings
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_predictions, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.title('Actual vs Predicted Ratings')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.show()