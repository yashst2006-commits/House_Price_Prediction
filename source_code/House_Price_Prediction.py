import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error , mean_squared_error

LR_model = LinearRegression()
DT_model = DecisionTreeRegressor()
RF_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)


df = pd.read_csv("House_Price_Prediction/dataset/house_prices_dataset.csv")
#df = df.drop("Id", axis=1)
#df = pd.get_dummies(
#   df,
#   columns=["Location", "Condition", "Parking"],
#   drop_first=True
#)

X = df.drop('price', axis=1)
Y = df["price"]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)



LR_model.fit(X_train, Y_train)
Y_pred_LR = LR_model.predict(X_test)
mae_LR = mean_absolute_error(Y_test , Y_pred_LR)
rmse_LR = np.sqrt(mean_squared_error(Y_test, Y_pred_LR))
print("LR MAE:", mae_LR)
print("LR RMSE:", rmse_LR)


DT_model.fit(X_train, Y_train)
Y_pred_DT = DT_model.predict(X_test)
mae_DT = mean_absolute_error(Y_test, Y_pred_DT)
rmse_DT = np.sqrt(mean_squared_error(Y_test, Y_pred_DT))
print("DT MAE:", mae_DT)
print("DT RMSE:", rmse_DT)


RF_model.fit(X_train, Y_train)
Y_pred_RF = RF_model.predict(X_test)
mae_RF = mean_absolute_error(Y_test, Y_pred_RF)
rmse_RF = np.sqrt(mean_squared_error(Y_test, Y_pred_RF))
print("RF MAE:", mae_RF)
print("RF RMSE:", rmse_RF)

#plt.scatter(Y_test, Y_pred_LR, alpha=0.6)
#min_price_LR = min(Y_test.min(), Y_pred_LR.min())
#max_price_LR = max(Y_test.max(), Y_pred_LR.max())
#plt.plot([min_price_LR, max_price_LR], [min_price_LR, max_price_LR])
#plt.xlabel("Actual Price")
#plt.ylabel("Predicted Price")
#plt.title("Actual vs Predicted Prices (Linear Regression)")
#plt.savefig("House_Price_Prediction/images/LR_Graph.png")


#plt.scatter(Y_test, Y_pred_DT, alpha=0.6)
#min_price_DT = min(Y_test.min(), Y_pred_DT.min())
#max_price_DT = max(Y_test.max(), Y_pred_DT.max())
#plt.plot([min_price_DT, max_price_DT], [min_price_DT, max_price_DT])
#plt.xlabel("Actual Price")
#plt.ylabel("Predicted Price")
#plt.title("Actual vs Predicted Prices (Decision Tree)")
#plt.savefig("House_Price_Prediction/images/DT_Graph.png")

plt.scatter(Y_test , Y_pred_RF , alpha=0.6)
min_price_RF = min(Y_test.min() , Y_pred_RF.min())
max_price_RF = max(Y_test.max() , Y_pred_RF.max())
plt.plot([min_price_RF, max_price_RF], [min_price_RF, max_price_RF])
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices (Random Forest)")
plt.savefig("House_Price_Prediction/images/RF_Graph.png")