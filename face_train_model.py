from data_loader import load_and_process_data
from face_model import build_face_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

(X_train, X_test, y_train, y_test), class_names = load_and_process_data('data')

model = build_face_model(X_train.shape[1:], len(class_names))

es = EarlyStopping(patience=10, restore_best_weights=True)
mc = ModelCheckpoint('saved_model/best_face_model.h5', save_best_only=True)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=50, batch_size=16, callbacks=[es, mc])

# Plot
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

# Save final
model.save('saved_model/final_face_model.h5')


#UnderConstruction Please wait for the final model
