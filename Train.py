from Dataset import loadTrain
from Dataset import loadTest
from Dataset import generator
import math
import keras
import matplotlib.pyplot as plt

BATCH_SIZE  = 3
EPOCHS = 5

x_train , y_train = loadTrain()
x_test, y_test = loadTest()

train_gen = generator(x_train, y_train, BATCH_SIZE)
test_gen = generator(x_test, y_test, x_test.shape[0])

N = x_train.shape[0]
steps_per_epoch = math.ceil(N / BATCH_SIZE)
print("INFO: Total datas = {}".format(N))
print("INFO: Step per epoch = {}".format(steps_per_epoch))

model = keras.applications.resnet50.ResNet50(weights=None, classes=3)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, validation_data=test_gen, validation_steps=1, epochs=EPOCHS)

# 学習と検証の正解率・損失の推移を取得
acc = history.history['acc']          # 訓練データの正解率
val_acc = history.history['val_acc']  # 検証データの正解率
loss = history.history['loss']        # 訓練データの損失率
val_loss = history.history['val_loss']# 検証データの損失率

# 過学習が発生していないかをグラフで確認する方法
# → 訓練データの正解率 > 検証データの正解率で、正解率が大きく離れていれば過学習している
# → 大きく離れていなくて近い正答率であっても、訓練データと検証データ以外のデータでは正答率が低い可能性がある(汎化性がなく過学習している)
#   その場合は訓練データと検証データとは別に用意したテストデータで検証を行う。

# グラフの縦軸(0~1), 横軸(1~5)
epochs = range(1, len(acc) + 1)

# 正解率をプロット
plt.plot(epochs, acc, 'b', label='[Training] accuracy')
plt.plot(epochs, val_acc, 'r', label='[Validation] accuracy')
plt.title('Accuracy') # グラフのタイトル
plt.legend() # 凡例のプロット

plt.figure()

# 損失率をプロット
plt.plot(epochs, loss, 'b', label='[Training] loss')
plt.plot(epochs, val_loss, 'r', label='[Validation] loss')
plt.title('Loss') # グラフのタイトル
plt.legend() # 凡例のプロット

# グラフの表示
plt.show()
