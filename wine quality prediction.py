import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
wine_data = pd.DataFrame({"fixed acidity": [7.4, 7.8, 7.8, 11.2, 7.4],
                          "volatile acidity": [0.70, 0.88, 0.76, 0.28, 0.70],
                          "citric acid": [0.00, 0.00, 0.04, 0.56, 0.00],
                          "residual sugar": [1.9, 2.6, 2.3, 1.9, 1.9],
                          "chlorides": [0.076, 0.098, 0.092, 0.075, 0.076],
                          "free sulfur dioxide": [11.0, 25.0, 15.0, 17.0, 11.0],
                          "total sulfur dioxide": [34.0, 67.0, 54.0, 60.0, 34.0],
                          "density": [0.9978, 0.9968, 0.9970, 0.9978, 0.9978],
                          "pH": [3.51, 3.20, 3.26, 3.26, 3.51],
                          "sulphates": [0.56, 0.68, 0.65, 0.65, 0.56],
                          "alcohol": [9.4, 9.8, 9.8, 9.8, 9.4],
                          "quality": [5, 5, 5, 5, 5]})
X_train, X_test, y_train, y_test = train_test_split(wine_data.drop("quality", axis=1), wine_data["quality"], test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error:", mse)
print(y_pred)
