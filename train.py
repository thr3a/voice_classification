import os

import joblib
import numpy as np
import opensmile
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# openSMILEを使用して音声ファイルから特徴量を抽出する関数
def extract_features(file_path):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    features = smile.process_file(file_path)
    return features.values.flatten()


# 音声ファイルとラベルを読み込む関数
def load_data(root_dir):
    X = []
    y = []
    for speaker_dir in os.listdir(root_dir):
        speaker_path = os.path.join(root_dir, speaker_dir)
        if os.path.isdir(speaker_path):
            for file in os.listdir(speaker_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(speaker_path, file)
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(speaker_dir)
    return np.array(X), np.array(y)


# モデルを学習する関数
def train_model(X, y):
    # データを訓練用とテスト用に分割（テストデータ20%）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ランダムフォレスト分類器を初期化して学習
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # テストデータで予測を行い、精度を計算
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"モデルの精度: {accuracy:.2f}")

    return clf


# モデルを保存する関数
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"モデルを {filename} に保存しました。")


# モデルをロードする関数
def load_model(filename):
    return joblib.load(filename)


if __name__ == "__main__":
    # 音声ファイルのルートディレクトリ
    root_dir = "./voices"

    # データの読み込みと特徴量抽出
    print("データを読み込み中...")
    X, y = load_data(root_dir)

    # モデルの学習
    print("モデルを学習中...")
    model = train_model(X, y)

    # モデルの保存
    model_filename = "voice_classification_model.joblib"
    save_model(model, model_filename)
