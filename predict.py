import pickle
import sys

from pyannote.audio import Inference, Model

# コマンドライン引数をチェック
if len(sys.argv) != 2:
    print("使用方法: python predict.py <音声ファイルのパス>")
    sys.exit(1)

# 音声ファイルのパスを取得
audio_file = sys.argv[1]

# pyannote/embeddingモデルを読み込む
model = Model.from_pretrained("pyannote/embedding", use_auth_token="YOUR_ACCESS_TOKEN")
inference = Inference(model, window="whole")

# 保存されたモデルを読み込む
with open("voices.pkl", "rb") as f:
    data = pickle.load(f)
    svm = data["svm"]
    le = data["label_encoder"]

# 新しい音声ファイルから埋め込みベクトルを抽出
embedding = inference(audio_file)

# embeddingを2D配列にリシェイプ (1 サンプル × N 特徴量)
embedding_reshaped = embedding.reshape(1, -1)

# 予測を行う
prediction = svm.predict(embedding_reshaped)
probabilities = svm.predict_proba(embedding_reshaped)

# 予測結果を人間が読める形式に変換
predicted_speaker = le.inverse_transform(prediction)[0]
confidence = probabilities[0][prediction[0]]

print(f"予測された話者: {predicted_speaker}")
print(f"信頼度: {confidence:.2f}")
