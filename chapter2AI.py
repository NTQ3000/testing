# import numpy as np
# import matplotlib.pyplot as plt
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import pandas as pd
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression

# # Dữ liệu mẫu
# # X: biến độc lập, y: biến phụ thuộc
# X = np.array([[30], [40], [50], [60], [70],[80]])
# y = np.array([60, 75, 105, 115, 140,150])

# # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=4)

# # Tạo mô hình hồi quy tuyến tính
# model = LinearRegression()

# # Huấn luyện mô hình
# model.fit(X_train, y_train)

# # Dự đoán
# y_pred = model.predict(X_test)

# # Hiển thị kết quả
# plt.scatter(X, y, color='blue', label='Dữ liệu gốc')
# plt.plot(X_test, y_pred, color='red', linewidth=2, label='Dự đoán')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('Mô hình hồi quy tuyến tính')
# plt.legend()
# plt.show()

# # In các hệ số
# print(f"Hệ số góc (slope): {model.coef_[0]}")
# print(f"Hệ số chặn (intercept): {model.intercept_}")
# print(f"phuong trinh la : y= {model.coef_[0]} x + {model.intercept_}")


import numpy as np
import matplotlib.pyplot as plt
x=np.array([30,40,50,60,70,80])

y= np.array([400,500,550,600,650,700,800,900])

from IPython.display import clear_output
m=-2
b=50
n=0.000001
for rep in range (120):
  dm=0
  db=0
  for i in range(8):
   dm += 2*n*(m*x[i]+b-y[i]*x[i])
   db += 2*n*(m*x[i]+b-y[i])
  m = m-dm/8
  b = b-db/8
  clear_output(wait=True)
  plt.plot(x,y,'o')
  plt.plot(x, m*x+b)
  plt.show()