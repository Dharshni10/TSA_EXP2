# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:
## AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

## ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
## PROGRAM:
```
Name : Dharshni V M
Register : 212223240029

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('gold.csv')
data.head()
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
resampled_data = data['EURO (PM)'].resample('Y').sum().to_frame()
resampled_data.head()
resampled_data.index = resampled_data.index.year
resampled_data.reset_index(inplace=True)
resampled_data.rename(columns={'Date': 'Year'}, inplace=True)
resampled_data.head()
years = resampled_data['Year'].tolist()
Euro = resampled_data['EURO (PM)'].tolist()

A - LINEAR TREND ESTIMATION

X = [i - years[len(years) // 2] for i in years]
x2 = [i ** 2 for i in X]
xy = [i * j for i, j in zip(X, Euro)]
n = len(years)
b = (n * sum(xy) - sum(Euro) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(Euro) - b * sum(X)) / n
linear_trend = [a + b * X[i] for i in range(n)]

B- POLYNOMIAL TREND ESTIMATION

x3 = [i ** 3 for i in X]
x4 = [i ** 4 for i in X]
x2y = [i * j for i, j in zip(x2, Euro)]
coeff = [[len(X), sum(X), sum(x2)],
         [sum(X), sum(x2), sum(x3)],
         [sum(x2), sum(x3), sum(x4)]]
Y = [sum(Euro), sum(xy), sum(x2y)]
A = np.array(coeff)
B = np.array(Y)
solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution
poly_trend = [a_poly + b_poly * X[i] + c_poly * (X[i] ** 2) for i in range(n)]

print(f"Linear Trend: y={a:.2f} + {b:.2f}x")
print(f"\nPolynomial Trend: y={a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")

resampled_data['Linear Trend'] = linear_trend
resampled_data['Polynomial Trend'] = poly_trend

resampled_data.set_index('Year',inplace=True)

resampled_data['EURO (PM)'].plot(kind='line',color='blue',marker='o') 
resampled_data['Linear Trend'].plot(kind='line',color='black',linestyle='--')

resampled_data['EURO (PM)'].plot(kind='line',color='blue',marker='o')
resampled_data['Polynomial Trend'].plot(kind='line',color='black',marker='o')
```

## OUTPUT

### TREND EQUATION 

![Equation](https://github.com/user-attachments/assets/b188de0a-e70c-42fe-a4f2-f3392de7439f)

### A - LINEAR TREND ESTIMATION

![Linear](https://github.com/user-attachments/assets/760705bc-3964-4408-832c-309745fac9fc)

### B- POLYNOMIAL TREND ESTIMATION

![Polynomial](https://github.com/user-attachments/assets/cc704dd0-f700-48d3-a794-77182d5926e1)

## RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
