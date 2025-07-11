import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

st.title("FFNN Training & Evaluation with Fusion Features")

# File uploaders
vgg_csv = st.file_uploader("Upload VGG19 features CSV", type=['csv'])
resnet_csv = st.file_uploader("Upload ResNet101 features CSV", type=['csv'])
handcrafted_csv = st.file_uploader("Upload Handcrafted features CSV", type=['csv'])
labels_csv = st.file_uploader("Upload Labels CSV", type=['csv'])
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

def build_ffnn(input_dim, num_classes):
    model = Sequential([
        Dense(2048, input_shape=(input_dim,)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(1024),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.4),

        Dense(512),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.3),

        Dense(256, activation='selu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(128, activation='selu'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])
    return model

if st.button("Start Training & Evaluation"):
    if vgg_csv and resnet_csv and handcrafted_csv and labels_csv:
        # Load data
        vgg_features = load_csv(vgg_csv)
        resnet_features = load_csv(resnet_csv)
        handcrafted_features = load_csv(handcrafted_csv)
        labels_df = load_csv(labels_csv)

        # Ensure order matches
        filenames = labels_df['filename'].values
        vgg_features = vgg_features.set_index('filename').loc[filenames].reset_index(drop=True)
        resnet_features = resnet_features.set_index('filename').loc[filenames].reset_index(drop=True)
        handcrafted_features = handcrafted_features.set_index('filename').loc[filenames].reset_index(drop=True)

        # Prepare splits
        y = labels_df['label'].values
        sets = labels_df['set'].values
        num_classes = len(np.unique(y))
        y_cat = to_categorical(y, num_classes)

        # Fusion 
        features_vgg_handcrafted = np.concatenate([vgg_features.drop(['filename', 'label', 'set'], axis=1).values,
                                                   handcrafted_features.drop(['filename', 'label', 'set'], axis=1).values], axis=1)
        features_resnet_handcrafted = np.concatenate([resnet_features.drop(['filename', 'label', 'set'], axis=1).values,
                                                      handcrafted_features.drop(['filename', 'label', 'set'], axis=1).values], axis=1)
        features_all = np.concatenate([vgg_features.drop(['filename', 'label', 'set'], axis=1).values,
                                       resnet_features.drop(['filename', 'label', 'set'], axis=1).values,
                                       handcrafted_features.drop(['filename', 'label', 'set'], axis=1).values], axis=1)

        # Standardize
        scaler = StandardScaler()
        features_vgg_handcrafted = scaler.fit_transform(features_vgg_handcrafted)
        features_resnet_handcrafted = scaler.fit_transform(features_resnet_handcrafted)
        features_all = scaler.fit_transform(features_all)

        # Split by 'set'
        train_idx = np.where(set == 'train')[0]
        eval_idx = np.where((set == 'val') | (set == 'test'))[0]

        # Prepare train/eval sets
        X_train = features_vgg_handcrafted[train_idx]
        y_train = y_cat[train_idx]
        X_eval = features_vgg_handcrafted[eval_idx]
        y_eval = y_cat[eval_idx]
        y_eval_labels = y[eval_idx]

        # Define optimizer and callbacks ONCE
        optimizer = Adam(learning_rate=0.0001)
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-7, verbose=1)

        # FFNN Training & Evaluation for VGG19 + Handcrafted
        st.subheader("FFNN Training (VGG19 + Handcrafted)")
        model_vgg_hc = build_ffnn(X_train.shape[1], num_classes)
        model_vgg_hc.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history_vgg_hc = model_vgg_hc.fit(
            X_train, y_train,
            epochs=100,
            batch_size=64,
            validation_split=0.1,
            verbose=1,
            callbacks=[early_stop, reduce_lr]
        )

        # Plot training history
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history_vgg_hc.history['accuracy'], label='Train Acc')
        ax[0].plot(history_vgg_hc.history['val_accuracy'], label='Val Acc')
        ax[0].set_title('Accuracy')
        ax[0].legend()
        ax[1].plot(history_vgg_hc.history['loss'], label='Train Loss')
        ax[1].plot(history_vgg_hc.history['val_loss'], label='Val Loss')
        ax[1].set_title('Loss')
        ax[1].legend()
        st.pyplot(fig)

        st.subheader("FFNN Evaluation (VGG19 + Handcrafted)")
        # FFNN EVALUATION
        eval_loss, eval_acc = model_vgg_hc.evaluate(X_eval, y_eval, batch_size=32, verbose=1)
        st.write(f"Loss: {eval_loss:.4f}")
        st.write(f"Accuracy: {eval_acc:.4f}")

        # Predictions and metrics
        y_pred = np.argmax(model_vgg_hc.predict(X_eval, batch_size=32), axis=1)
        st.write("Classification Report:")
        st.text(classification_report(y_eval_labels, y_pred))
        st.write("Confusion Matrix:")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_eval_labels, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax2)
        st.pyplot(fig2)

        # FFNN Training & Evaluation for ResNet101 + Handcrafted
        st.subheader("FFNN Training (ResNet101 + Handcrafted)")
        X_train_rh = features_resnet_handcrafted[train_idx]
        X_eval_rh = features_resnet_handcrafted[eval_idx]
        model_resnet_hc = build_ffnn(X_train_rh.shape[1], num_classes)
        history_resnet_hc = model_resnet_hc.fit(X_train_rh, y_train, epochs=50, batch_size=64, validation_split=0.1, verbose=1)

        # Plot training history
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history_resnet_hc.history['accuracy'], label='Train Acc')
        ax[0].plot(history_resnet_hc.history['val_accuracy'], label='Val Acc')
        ax[0].set_title('Accuracy')
        ax[0].legend()
        ax[1].plot(history_resnet_hc.history['loss'], label='Train Loss')
        ax[1].plot(history_resnet_hc.history['val_loss'], label='Val Loss')
        ax[1].set_title('Loss')
        ax[1].legend()
        st.pyplot(fig)

        st.subheader("FFNN Evaluation (ResNet101 + Handcrafted)")
        # FFNN EVALUATION
        eval_loss, eval_acc = model_resnet_hc.evaluate(X_eval_rh, y_eval, batch_size=32, verbose=1)
        st.write(f"Loss: {eval_loss:.4f}")
        st.write(f"Accuracy: {eval_acc:.4f}")

        # Predictions and metrics
        y_pred_rh = np.argmax(model_resnet_hc.predict(X_eval_rh, batch_size=32), axis=1)
        st.write("Classification Report:")
        st.text(classification_report(y_eval_labels, y_pred_rh))
        st.write("Confusion Matrix:")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_eval_labels, y_pred_rh), annot=True, fmt="d", cmap="Blues", ax=ax2)
        st.pyplot(fig2)

        # FFNN Training & Evaluation for VGG19 + ResNet101 + Handcrafted
        st.subheader("FFNN Training (VGG19 + ResNet101 + Handcrafted)")
        X_train_all = features_all[train_idx]
        X_eval_all = features_all[eval_idx]
        model_all = build_ffnn(X_train_all.shape[1], num_classes)
        history_all = model_all.fit(X_train_all, y_train, epochs=50, batch_size=64, validation_split=0.1, verbose=1)

        # Plot training history
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history_all.history['accuracy'], label='Train Acc')
        ax[0].plot(history_all.history['val_accuracy'], label='Val Acc')
        ax[0].set_title('Accuracy')
        ax[0].legend()
        ax[1].plot(history_all.history['loss'], label='Train Loss')
        ax[1].plot(history_all.history['val_loss'], label='Val Loss')
        ax[1].set_title('Loss')
        ax[1].legend()
        st.pyplot(fig)

        st.subheader("FFNN Evaluation (VGG19 + ResNet101 + Handcrafted)")
        # FFNN EVALUATION
        eval_loss, eval_acc = model_all.evaluate(X_eval_all, y_eval, batch_size=32, verbose=1)
        st.write(f"Loss: {eval_loss:.4f}")
        st.write(f"Accuracy: {eval_acc:.4f}")

        # Predictions and metrics
        y_pred_all = np.argmax(model_all.predict(X_eval_all, batch_size=32), axis=1)
        st.write("Classification Report:")
        st.text(classification_report(y_eval_labels, y_pred_all))
        st.write("Confusion Matrix:")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_eval_labels, y_pred_all), annot=True, fmt="d", cmap="Blues", ax=ax2)
        st.pyplot(fig2)

        st.success("Training and evaluation complete!")
    else:
        st.warning("Please upload all required CSV files.")