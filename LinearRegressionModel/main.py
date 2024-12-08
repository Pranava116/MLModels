import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("salary.csv")

def Gradient_Descent(m, b, points, L):
    m1 = 0
    b1 = 0
    n = len(points)
    for i in range(n):
        x = data.iloc[i].YearsExperience
        y = data.iloc[i].Salary
        m1 += -(2/n) * (y - m*x - b )*(x)
        b1 += -(2/n) * (y - m*x - b)
    m_final = m - L*m1
    b_final = m-L*b1
    return m_final, b_final
    
m = 0
b = 0
L = 0.0001
iterations = 400

for i in range(iterations):
    m, b = Gradient_Descent(m, b,data, L)


print(m, b)
plt.scatter(data.YearsExperience, data.Salary)
plt.plot(list(range(0, 25)), [ m * x + b for x in range(0, 25)])
plt.show()

        
