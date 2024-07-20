import os
import pickle

import librosa
from pyannote.audio import Inference, Model
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# pyannote/embeddingモデルを読み込む
model = Model.from_pretrained("pyannote/embedding", use_auth_token="YOUR_ACCESS_TOKEN")
inference = Inference(model, window="whole")

# 話者の音声ファイルが格納されているディレクトリ
voices_dir = "./voices"

embeddings = []
labels = []

# 各話者のディレクトリをループ
for speaker in os.listdir(voices_dir):
    speaker_dir = os.path.join(voices_dir, speaker)
    if os.path.isdir(speaker_dir):
        # 各音声ファイルをループ
        for file in os.listdir(speaker_dir):
            if file.endswith(".wav"):
                file_path = os.path.join(speaker_dir, file)
                print(file_path)
                duration = librosa.get_duration(path=file_path)
                # 一定秒数未満の音声ファイルはスキップ
                threshold_seconds = 2
                if duration < threshold_seconds:
                    print(
                        f"Skipping {file_path}: Duration is less than {threshold_seconds} seconds ({duration:.2f}s)"
                    )
                    continue
                # 音声ファイルから埋め込みベクトルを抽出
                embedding = inference(file_path)
                embeddings.append(embedding.squeeze())
                labels.append(speaker)

# ラベルをエンコード
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

# SVMモデルを訓練
svm = SVC(kernel="rbf", probability=True)
svm.fit(embeddings, encoded_labels)

# モデルと関連情報を保存
with open("voices.pkl", "wb") as f:
    pickle.dump({"svm": svm, "label_encoder": le}, f)

print("saved as voices.pkl")
