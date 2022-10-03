import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from datasist.structdata import detect_outliers
from scipy import stats
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

data = pd.read_excel('Data_Train.xlsx')
display(data)
data.info()

data.dropna(subset=['Total_Stops','Route'], inplace = True)
sns.boxplot(data = data, x='Price')

#Convert Date_of_journey to datetime
data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'], format ='%d/%m/%Y')

#Appending month, day and weekday to the dataset
data['Month'] = data['Date_of_Journey'].dt.month
data['WeekDay'] = data['Date_of_Journey'].dt.day_name()
data['Day'] = data['Date_of_Journey'].dt.day

data['Weekend'] = data['WeekDay'].apply(lambda x : 1 if x == 'Sunday' else 0)

#Convert duration to seconds
def convert_dur(duration):
    try:
        hour_to_second = int(duration.split(' ')[0].replace('h',''))* 60 * 60
    except:
        hour_to_second = 0
    try:
        minute_to_second = int(duration.split(' ')[1].replace('m','')) * 60
    except:
        minute_to_second = 0
    return hour_to_second + minute_to_second

data['Duration'] = data['Duration'].apply(lambda x:convert_dur(x))

display(data)

data['Month'] = data['Month'].astype(str)
data['Day'] = data['Day'].astype(str)
data['Weekend'] = data['Weekend'].astype(str)
data.info()

data['Additional_Info'].unique()
data['Additional_Info'].replace('No Info','No info', inplace=True)

c1 = sns.countplot(data = data, y='Airline')
c1.set(title='Most used airline')
c1.figure.savefig("mostUsedAirline.jpg")

plt.figure(figsize=(10,10))
c12 = plt.pie(x=data.groupby('Airline')['Price'].count(), autopct='%1.2f%%')
plt.legend(data.groupby('Airline')['Price'].count().index)
plt.savefig("mostUsedAirlinePie.svg")

c2 = sns.countplot(data = data, x = 'Source')
c2.set(title='Number of takeoffs in cities')
c2.figure.savefig("NumberOfTakeoffsInCities.svg")

c3 = sns.countplot(data = data, x='Destination')
c3.set(title='The city where the plane landed')
c3.figure.savefig("TheCityWhereThePlaneLanded.svg")

c4 = sns.countplot(data = data, x='Total_Stops')
c4.set(title='Number of stops in each flight')
c4.figure.savefig("NumberOfStopsInEachFlight.svg")

c5 = sns.countplot(data = data, x='Day', hue='Month')
c5.set(title='number of flights of days in a month by month')
c5.figure.savefig("numberOfFlightsOfDaysInAMonthByMonth.svg")

c7 = sns.barplot(data=data, y=data.groupby('Additional_Info')['Price'].mean().index, x=data.groupby('Additional_Info')['Price'].mean().values)
c7.set(title='average fare for each class of seats')
c7.figure.savefig("AverageFareForEachClassOfSeats.svg", bbox_inches = 'tight')

price_by_weekend = pd.DataFrame(data.groupby('Weekend')['Price'].mean())
price_by_weekend.reset_index(inplace=True)
price_by_weekend['W'] = price_by_weekend['Weekend'].apply(lambda x : 'Weekend' if x== '1' else 'WeekDay')
c6 = sns.barplot(data = price_by_weekend, x='W', y='Price')
c6.set(title='Ticket price between weekday and weekend')
c6.figure.savefig("TicketPriceBetweenWeekdayAndWeekend.svg")

plt.figure(figsize = (15,8))
c8 = sns.barplot(data=data, x='Airline', y='Price',hue='WeekDay')
plt.xticks(rotation = 60)
c8.set(title='Fares of each airline by day of the week')
c8.figure.savefig("FaresOfEachAirlineByDayOfTheWeek.svg",bbox_inches = 'tight')

plt.figure(figsize = (15,8))
c9 = sns.barplot(data=data, x='Airline', y='Price',hue='Month')
plt.xticks(rotation = 60)
c9.set(title='Fares of each airline by month')
c9.figure.savefig("FaresOfEachAirlineByMonth.svg", bbox_inches = 'tight')

plt.figure(figsize = (15,8))
c10 = sns.barplot(data=data, x='Airline', y='Price')
plt.xticks(rotation = 60)
c10.set(title='Fares of each airline')
c10.figure.savefig("FaresOfEachAirline.jpg")

data_n = data.copy()
data_n.head()
outliers_indices = detect_outliers(data_n,0,['Price'])
data_n.drop(outliers_indices,inplace = True)
len(outliers_indices)

#correlation test
q1, q2, q3 = data_n.Price.quantile(0.25), data_n.Price.quantile(0.5), data_n.Price.quantile(0.75)
def price_order(price):
    if price < q1:
        return 1
    elif price >=q1 and price <q2:
        return 2
    elif price >=q2 and price < q3:
        return 3
    else :
        return 4
data_n['price_ordinal']=data_n.Price.apply(lambda x : price_order(x))
data_n.head()

col=['Source', 'Destination', 'Total_Stops', 'Additional_Info', 'Month', 'Day', 'WeekDay', 'Airline', 'Weekend']
for c in col:
    print(f'Kiểm định tương quan giữa {c} và Price')
    r, pvalue = stats.spearmanr(data_n[c], data_n.price_ordinal)
    print("r : ",r, "p : ",pvalue)
    
sns.lmplot(data=data_n, x='Duration', y='Price')
plt.savefig("DurationVsPrice.svg")

x = data_n.drop(['Date_of_Journey','Route','Dep_Time', 'Arrival_Time', 'Month' , 'Price','price_ordinal'], axis = 1)
y = data_n['Price']

x = pd.get_dummies(x,columns=['Airline', 'Source', 'Destination', 'Total_Stops', 'WeekDay', 'Additional_Info', 'Day','Weekend'], drop_first=True)
# x = pd.get_dummies(x, drop_first=True)
x

scaler = StandardScaler()
scaler.fit(x)
x_new = scaler.transform(x)

x_train,x_test,y_train,y_test = train_test_split(x_new,y,test_size=0.2,random_state=0,shuffle = False)

def performance(model,x_train,y_train,y_pred,y_test):
    
    print('Training Score:',model.score(x_train,y_train))
    print('Testing Score:',r2_score(y_test,y_pred))
    print('Other Metrics In Testing Data: ')
    print('MSE:',mean_squared_error(y_test,y_pred))
    print('MAE:',mean_absolute_error(y_test,y_pred))
 
#Fitting the model
lr = LinearRegression()
lr.fit(x_train,y_train)
#The predicted data
lr_pred = lr.predict(x_test)
performance(lr,x_train,y_train,lr_pred,y_test)

#Fitting the model
ridge = Ridge(alpha = 1)
ridge.fit(x_train,y_train)
#The predicted data
ridge_pred = ridge.predict(x_test)
#The performance
performance(ridge,x_train,y_train,ridge_pred,y_test)

#Fitting the model
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
#The predicted data
dt_pred = dt.predict(x_test)
#The performance
performance(dt,x_train,y_train,dt_pred,y_test)

#Fitting the model
rf = RandomForestRegressor()
rf.fit(x_train,y_train)
#The predicted data
rf_pred = rf.predict(x_test)
#The performance
performance(rf,x_train,y_train,rf_pred,y_test)

#Fitting the model
xgb = XGBRegressor()
xgb.fit(x_train,y_train)
#The predicted data
xgb_pred = xgb.predict(x_test)
#The performance
performance(xgb,x_train,y_train,xgb_pred,y_test)

#Actual data vs Predict data
plt.scatter(xgb_pred,y_test,c='blue',marker='o',s=25)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],c='black',lw=2)
plt.xlabel('Predicted Data',c='red')
plt.ylabel('Actual Data',c='red')
plt.title('Predicted Data VS Actual Data',c='red')
plt.savefig('preVsActual.svg')
