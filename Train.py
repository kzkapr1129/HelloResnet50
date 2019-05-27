from Dataset import loadTrain
from Dataset import loadTest
from Dataset import generator
from Dataset import NUM_CLASSES
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

# start trainning
model = keras.applications.resnet50.ResNet50(weights=None, classes=NUM_CLASSES)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, validation_data=test_gen, validation_steps=1, epochs=EPOCHS)

# save model & weights
json_string = model.to_json()
open('resnet50_model.json', 'w').write(json_string)
model.save_weights('resnet50_weights.h5')

# show graph
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='[Training] accuracy')
plt.plot(epochs, val_acc, 'r', label='[Validation] accuracy')
plt.title('Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='[Training] loss')
plt.plot(epochs, val_loss, 'r', label='[Validation] loss')
plt.title('Loss')
plt.legend()

plt.show()
