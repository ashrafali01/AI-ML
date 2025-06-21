import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def plot_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.title('Loss')

    plt.show()

def evaluate_model(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(classification_report(y_test, y_pred_classes, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def train_model(model, X_train, y_train, X_val, y_val, save_path="saved_model/best_model.h5"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    es = EarlyStopping(patience=10, restore_best_weights=True, verbose=1)
    mc = ModelCheckpoint(save_path, save_best_only=True, verbose=1)

    history = model.fit(X_train, y_train, 
                        epochs=50,
                        validation_data=(X_val, y_val),
                        callbacks=[es, mc],
                        batch_size=16)
    
    plot_history(history)
    return model
