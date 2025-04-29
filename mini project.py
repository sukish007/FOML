import tensorflow as tf
import numpy as np

# XOR input and output
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Build a tiny neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_shape=(2,), activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=500, verbose=0)

# Evaluate on inputs
predictions = model.predict(X).round()

# Display results
for i, input in enumerate(X):
    print(f"Input: {input}, Predicted: {int(predictions[i][0])}, True: {int(y[i][0])}")
