# Các thư viện dùng cho file train.py và test.py
pandas              1.0.3
numpy               1.18.2
xgboost             1.1.1
sklearn             0.23.2

# Cú pháp dòng lệnh chạy file train.py
python <đường-dẫn-tới-train.py> <đường-dẫn-tới-X_train.csv> <đường-dẫn-tới-Y_train.csv> <đườn-dẫn-tới-output.model>
(thay python bằng python3 nếu máy có nhiều phiên bản python)

# Cú pháp dòng lệnh chạy file test.py
python <đường-dẫn-tới-test.model> <đường-dẫn-tới-X.csv> <đường-dẫn-tới-predictY.csv>
(thay python bằng python3 nếu máy có nhiều phiên bản python)

2 file train.py và test.py cần đặt chung thư mục vì train.py có xuất ra file category.encoder để lưu dữ liệu về các thuộc tính trong X_train.csv
