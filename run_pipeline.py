from src.data_loader import load_data
from src.model import build_model
from src.train import train_model, evaluate_model

DATA_DIR = 'C:\Main\Dirilis 2025\FaceClassification\data'  # your data folder
IMG_SIZE = (128, 128)

# 1️⃣ Load Data
(X_train, X_test, y_train, y_test), class_names = load_data(DATA_DIR, img_size=IMG_SIZE)

# 2️⃣ Build Model
input_shape = X_train.shape[1:]
num_classes = len(class_names)

model = build_model(input_shape, num_classes)

# 3️⃣ Train
model = train_model(model, X_train, y_train, X_test, y_test)

# 4️⃣ Evaluate
evaluate_model(model, X_test, y_test, class_names)

# 5️⃣ Save final model
model.save("saved_model/final_face_classifier.h5")
