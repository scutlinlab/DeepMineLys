import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import math
from tensorflow.keras.models import Model, model_from_json

# 禁用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def read_file(filepath):
    """
    Read the input sequence feature file. 
    The required format:
    The first column is the sequence name, and the second and later columns are the encoded sequence features.
    """
    data = pd.read_csv(filepath, header=None)
    features = data.iloc[:, 1:].values.astype('float64')
    sequence_names = data.iloc[:, 0].values
    return features, sequence_names

def sigmoid_function(z):
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            z[i][j]=(1/(1 + math.exp(-z[i][j])))
    return z

def transform_data(array):
    """
    Process input sequence features
    """
    unirep_features = array[:, :1900]
    #phy_features = normalize_features(array[:, 1900:1904])
    phy_features = sigmoid_function(array[:, 1900:1904])
    unirep_array = unirep_features.reshape(-1, 38, 25, 2).astype('float64')
    phy_array = phy_features.reshape(-1, 2, 2, 1).astype('float64')

    return unirep_array, phy_array

def load_model(model_dir, model_name, weights_name):
    """
    Load the model's JSON file and weights
    
    """
    with open(os.path.join(model_dir, model_name), 'r') as json_file:
        model_json = json_file.read()

    model = model_from_json(model_json)
    model.load_weights(os.path.join(model_dir, weights_name))
    print(f"Loaded model {model_name} weights from {weights_name}")
    return model

def main(input_path, model_dir, output_path):
    # 读取并处理数据
    data, item_names = read_file(input_path)
    unirep_data, phy_data = transform_data(data)

    # 加载模型
    unirep_model = load_model(model_dir, 'Unirep_classifier.json', 'Unirep_best_weights.hdf5')
    phy_model = load_model(model_dir, 'PHY_classifier.json', 'PHY_best_weights.hdf5')
    mix_model = load_model(model_dir, 'Last_classifier.json', 'MIX_best_weights.hdf5')

    # 创建用于提取中间层特征的模型
    unirep_feature_extractor = Model(inputs=unirep_model.input, outputs=unirep_model.layers[-2].output)
    phy_feature_extractor = Model(inputs=phy_model.input, outputs=phy_model.layers[-2].output)

    # 生成特征
    unirep_features = unirep_feature_extractor.predict(unirep_data)
    phy_features = phy_feature_extractor.predict(phy_data)

    # 合并特征并进行预测
    combined_features = np.c_[unirep_features, phy_features].reshape(-1, 1, 128, 1)
    predictions = mix_model.predict(combined_features)

    # 处理预测结果
    labels = np.argmax(predictions, axis=1)
    results = np.c_[item_names, labels, predictions]

    # 输出结果
    np.savetxt(output_path, results, fmt="%s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lysin Prediction")
    parser.add_argument('--input_path', type=str, required=True, help="Input path of the sequence feature file")
    parser.add_argument('--model_path', type=str, required=True, help="The path of the saved model")
    parser.add_argument('--output_path', type=str, required=True, help="Output path for the results")

    args = parser.parse_args()
    main(args.input_path, args.model_path, args.output_path)
