import cv2
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

model_attention = load_model('model_attention(1-20).h5')

Test_Dir = 'test.csv'
test_data = pd.read_csv(Test_Dir)

timag = []
for i in range(0,9):
    timg = test_data['Image'][i].split(' ')
    timg = ['0' if x == '' else x for x in timg]
    
    timag.append(timg)

timage_list = np.array(timag,dtype = 'float')
timage_list = timage_list/255.
X_test = timage_list.reshape(-1,96,96,1)

pred = model_attention.predict(X_test)

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2]*48+48, y[1::2]*48+48, marker='o', s=10, c='red')

fig = plt.figure(figsize=(10, 7))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(9):
    axis = fig.add_subplot(3, 3, i+1, xticks=[], yticks=[])
    plot_sample(X_test[i], pred[i], axis)

plt.show()