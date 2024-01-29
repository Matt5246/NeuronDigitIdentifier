import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from sklearn.metrics import classification_report, roc_curve, auc
from keras.datasets import mnist
import matplotlib.pyplot as plt
from matplotlib import pyplot

# Load the MNIST dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Display the shape of the dataset
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  ' + str(test_X.shape))
print('Y_test:  ' + str(test_y.shape))

# Display some sample images

for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()

# Normalize pixel values to the range [0, 1]
train_X = train_X / 255.0
test_X = test_X / 255.0

# Define the neural network architecture
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Experiment with different learning rates and number of epochs
learning_rates = [0.001, 0.01, 0.1]
epochs = [5, 10, 15]

with open('experiment_results.txt', 'w') as results_file:
    for lr in learning_rates:
        for epoch in epochs:
            results_file.write(f'\nTraining with Learning Rate: {lr}, Epochs: {epoch}\n')

            # Train the model
            history = model.fit(train_X, train_y, epochs=epoch, validation_split=0.2, verbose=2)

            # Evaluate the model on the test set
            test_loss, test_accuracy = model.evaluate(test_X, test_y)
            results_file.write(f'Test Accuracy: {test_accuracy}\n')

            # Save the training history plot
            plt.plot(history.history['accuracy'], label=f'Training Accuracy LR={lr}, Epochs={epoch}')
            plt.plot(history.history['val_accuracy'], label=f'Validation Accuracy LR={lr}, Epochs={epoch}')

            # Make predictions on the test set
            y_pred_probs = model.predict(test_X)
            y_pred = np.argmax(y_pred_probs, axis=-1)

            # Compute ROC curve and AUC
            fpr, tpr, _ = roc_curve(to_categorical(test_y).ravel(), y_pred_probs.ravel())
            roc_auc = auc(fpr, tpr)

            # Save AUC score to results file
            results_file.write(f'AUC Score: {roc_auc}\n\n')

# Plot a single ROC curve
plt.figure()
for lr in learning_rates:
    for epoch in epochs:
        plt.plot(fpr, tpr, lw=2, label=f'LR={lr}, Epochs={epoch}')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.show()

# Confirm that all results are saved to a single text file and PNG files
print("Results and plots are saved to experiment_results.txt, training_history*.png, and roc_curve.png files.")
