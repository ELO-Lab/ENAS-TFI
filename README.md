# Efficiency Enhancement of Evolutionary Neural Architecture Search via Training-Free Initialization
## Installation
Note: Khuyến khích sử dụng pip để cài dặt
- python3 (>= 3.5)
- matplotlib (version 3.0.3)\
`python -m pip install -U matplotlib`
- sklearn (version 0.21.2)\
`pip install -U scikit-learn` nếu sử dụng pip\
`conda install scikit-learn` nếu sử dụng conda
- joblib (>= 0.11)\
`pip install joblib`
- numpy (>= 1.11.0)\
`python -m pip install numpy`
- cv2 (version 4.1.0)\
`pip install opencv-python` nếu chỉ sử dụng module chính\
`pip install opencv-contrib-python` nếu muốn sử dụng đầy đủ module, chi tiết xem thêm ở [OpenCV documentation](https://docs.opencv.org/master/)
- skimage (version 0.15.0)\
`pip install scikit-image` nếu sử dụng pip\
`conda install -c conda-forge scikit-image` nếu sử dụng conda
- os

- scipy (>= 0.17.0)
`pip install scipy`

## Hướng dẫn sử dụng
- Tải clone về máy.
- Để chung các mục folder `banhmi, cookie, donut, pizza, pretzel, sungbo` vào một folder (đặt tên là "train").
- Nếu trong file zip tải về không có file `clf_logistic.joblib, clf_svm.joblib` hay `labels.txt` thì:\
Chạy file `train-model-svm.py` nếu muốn tạo model dự đoán sử dụng thuật toán SVM.\
Chạy file `train-model-logistic` nếu muốn tạo model dự đoán sử dụng thuật toán Logistic Regression.
- Sau khi đã có model:\
Nếu muốn sử dụng model SVM để dự đoán thì chạy file `test-model-svm.py`.\
Nếu muốn sử dụng model Logistic Regression để dự đoán thì chạy file `test-model-logistic.py`.
- Khi chạy model để dự đoán, chương trình sẽ yêu cầu nhập đường dẫn tới ảnh cần dự đoán.\
Ví dụ: `C:\Users\FanKuan\Desktop\13.jpg`
- Sau khi chạy chương trình, chương trình sẽ trả về bức ảnh chứa loại bánh, tên loại bánh và tên thuật toán đã sử dụng để dự đoán.
