import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# === 参数设置 ===
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 29  # A-Z, del, nothing, space
EPOCHS = 8
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

# === 数据增强与加载 ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2]
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=False
)

# === 模型结构 ===
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# === 模型训练 ===
checkpoint = ModelCheckpoint("hand_sign_cnn_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)

model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[checkpoint]
)

# model.fit(
#     train_generator,
#     steps_per_epoch=3,           # 每轮只跑 5 个 batch（即 batch_size × 5 张图）
#     validation_data=test_generator,
#     validation_steps=1,
#     epochs=2,    # 轮数少也可以先调试用
#     callbacks=[checkpoint]
# )

# === 测试集评估 ===
loss, acc = model.evaluate(test_generator)
print(f"\nTest Accuracy: {acc:.4f}, Loss: {loss:.4f}")
