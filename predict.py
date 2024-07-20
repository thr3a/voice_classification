import joblib
import opensmile


# openSMILEを使用して音声ファイルから特徴量を抽出する関数
def extract_features(file_path):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    features = smile.process_file(file_path)
    return features.values.flatten()


# モデルをロードする関数
def load_model(filename):
    return joblib.load(filename)


# 新しい音声ファイルを分類する関数
def classify_new_audio(model, file_path):
    features = extract_features(file_path)
    prediction = model.predict([features])
    return prediction[0]


if __name__ == "__main__":
    model_filename = "voice_classification_model.joblib"
    loaded_model = load_model(model_filename)
    new_audio_file = "/Users/thr3a/works/docker-build-station/jupyterlab/voices/uemura_normal 2/uemura_normal_095.wav"
    prediction = classify_new_audio(loaded_model, new_audio_file)
    print(f"新しい音声ファイルの予測クラス: {prediction}")
