# Cnn_Mlp_Generator_Model.py

import os
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

# -----------------------
# 1. 自定义生成器类
# -----------------------
class ImageKeypointGenerator(Sequence):
    def __init__(self, image_root, keypoint_root, labels, batch_size=32, image_size=(128, 128), keypoint_dim=123, shuffle=True):
        self.image_root = image_root
        self.keypoint_root = keypoint_root
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.keypoint_dim = keypoint_dim
        self.shuffle = shuffle
        self.filepaths = self._load_filepaths()
        self.on_epoch_end()

    def _load_filepaths(self):
        filepaths = []
        for label_index, label in enumerate(self.labels):
            img_dir = os.path.join(self.image_root, label)
            kp_dir = os.path.join(self.keypoint_root, label)
            if not os.path.isdir(img_dir) or not os.path.isdir(kp_dir):
                continue
            for fname in os.listdir(img_dir):
                if fname.lower().endswith('.jpg'):
                    img_path = os.path.join(img_dir, fname)
                    kp_path = os.path.join(kp_dir, os.path.splitext(fname)[0] + '.npy')
                    filepaths.append((img_path, kp_path, label_index))
        return filepaths

    def __len__(self):
        return int(np.ceil(len(self.filepaths) / self.batch_size))

    def __getitem__(self, index):
        batch_paths = self.filepaths[index * self.batch_size:(index + 1) * self.batch_size]
        X_img, X_kp, y = [], [], []
        for img_path, kp_path, label_index in batch_paths:
            try:
                img = load_img(img_path, target_size=self.image_size)
                img = img_to_array(img) / 255.0
                keypoints = np.load(kp_path).flatten()
                if keypoints.shape[0] != self.keypoint_dim:
                    continue
                # if np.all(keypoints == 0):
                #     continue
                if np.all(keypoints == 0):
                    if 'test' in self.image_root.lower():
                        keypoints = np.full((self.keypoint_dim,), 0.5, dtype=np.float32)  # 用均值或 0.5 填充
                        print(f"⚠️ 测试集中保留无关键点: {img_path}")
                    else:
                        continue  # 训练集跳过
                X_img.append(img)
                X_kp.append(keypoints)
                y.append(label_index)
            except:
                continue
        return [np.array(X_img), np.array(X_kp)], to_categorical(y, num_classes=len(self.labels))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.filepaths)

# -----------------------
# 2. 模型构建
# -----------------------
def build_cnn_mlp_model(input_shape=(128, 128, 3), keypoint_dim=123, num_classes=29):
    # CNN 分支
    cnn_branch = models.Sequential([
        layers.Input(shape=input_shape),

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
    ])

    # MLP 分支
    input_kp = Input(shape=(keypoint_dim,))
    k = Dense(512, activation='relu')(input_kp)
    k = BatchNormalization()(k)
    k = Dropout(0.3)(k)

    k = Dense(256, activation='relu')(k)
    k = BatchNormalization()(k)
    k = Dropout(0.3)(k)

    k = Dense(128, activation='relu')(k)

    # 合并 CNN 和 MLP
    merged = Concatenate()([cnn_branch.output, k])
    output = Dense(num_classes, activation='softmax')(merged)

    model = Model(inputs=[cnn_branch.input, input_kp], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# -----------------------
# 3. 数据生成器与训练
# -----------------------
if __name__ == '__main__':
    labels = sorted(os.listdir('dataset/train'))
    image_size = (128, 128)
    keypoint_dim = 123
    batch_size = 32

    train_gen = ImageKeypointGenerator(
        image_root='dataset/train',
        keypoint_root='dataset/train_keypoints_123d',
        labels=labels,
        batch_size=batch_size,
        image_size=image_size,
        keypoint_dim=keypoint_dim
    )

    val_gen = ImageKeypointGenerator(
        image_root='dataset/test',
        keypoint_root='dataset/test_keypoints_123d',
        labels=labels,
        batch_size=batch_size,
        image_size=image_size,
        keypoint_dim=keypoint_dim,
        shuffle=False
    )

    model = build_cnn_mlp_model(input_shape=(128, 128, 3), keypoint_dim=keypoint_dim, num_classes=len(labels))
    model.fit(train_gen, validation_data=val_gen, epochs=5)
    model.save("cnn_mlp_generator_model.h5")
