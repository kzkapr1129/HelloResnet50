import numpy as np
import glob
import cv2

CLASSES = ['bar', 'foo', 'hoge']
NUM_CLASSES = len(CLASSES)
DIR_TRAIN = 'train'
DIR_TEST = 'test'

def preprocess(img):
    # img = cv2.resize(img, (224,224), interpolation = cv2.INTER_LINEAR)  # 拡大する場合
	img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)  # 縮小する場合
	return img

def loadImages(pathes):
    imgs = []
    for path in pathes:
        img = preprocess(cv2.imread(path))
        imgs.append(img)
    return imgs

def generator(x_train, y_train, num_batch):
	N = x_train.shape[0]
	while True:
		ridxes = np.random.permutation(N)
		for i in range(N-num_batch+1):
			idxes = ridxes[i:i+num_batch if i+num_batch < N else N]
			yield np.array(loadImages(x_train[idxes])), np.array(y_train[idxes])

def loadData(dir):
	x_data = []
	y_data = np.empty((0,len(CLASSES)), int)
	for cn in CLASSES:
		searchWords = dir + "/" + cn + "/" + "*.png"
		x = glob.glob(searchWords)
		x_data = np.hstack((x_data, x))
		y = np.zeros(len(CLASSES))
		y[CLASSES.index(cn)] = 1
		y_array = np.array([y for ii in range(len(x))])
		y_data = np.append(y_data, np.array(y_array), axis=0)
	return x_data, y_data

###
# 本関数は以下のフォルダ構成で保存されている画像データを読み込み、画像データと教師ラベルを返却します。
# root/
# ┣ Dataset.py
# ┗ train
#      ┣ bar ※ CLASSES[0]の文字列
#      ┃  ┗ *.png
#      ┣ foo ※ CLASSES[1]の文字列
#      ┃  ┗ *.png
#      ┗ hoge ※ CLASSES[2]の文字列
#            ┗ *.png
###
def loadTrain():
	return loadData(DIR_TRAIN)

###
# 本関数は以下のフォルダ構成で保存されている画像データを読み込み、画像データと教師ラベルを返却します。
# root/
# ┣ Dataset.py
# ┗ test
#      ┣ bar ※ CLASSES[0]の文字列
#      ┃  ┗ *.png
#      ┣ foo ※ CLASSES[1]の文字列
#      ┃  ┗ *.png
#      ┗ hoge ※ CLASSES[2]の文字列
#            ┗ *.png
###
def loadTest():
	return loadData(DIR_TEST)