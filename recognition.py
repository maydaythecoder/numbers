import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


train_new_model = not os.path.exists('handwritten_digits.model.keras')


if train_new_model:
    mnist = tf.keras.datasets.mnist  
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = tf.keras.utils.normalize(X_train, axis=1)  
    X_test = tf.keras.utils.normalize(X_test, axis=1)  
    model = tf.keras.models.Sequential()  
    model.add(tf.keras.layers.Flatten())  
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.leaky_relu))  
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.leaky_relu))  
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))  
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=3)
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss)
    print(val_acc)
    model.save('handwritten_digits.model.keras')
else:
    model = tf.keras.models.load_model('handwritten_digits.model.keras')  

image_number = 1
while os.path.isfile('digits/digit{}.png'.format(image_number)):
    try:
        img_raw = cv2.imread('digits/digit{}.png'.format(image_number))
        if img_raw is None:
            raise ValueError(f"Failed to load image digits/digit{image_number}.png")
        img = img_raw[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("The number is probably a {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap='binary')
        plt.show()
        image_number += 1

    except:
        print("Error reading image! Proceeding with next image...")
        image_number += 1
