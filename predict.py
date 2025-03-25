import torch
import torch.nn as nn
import numpy as np
from model import FeedForwardNN  # Ensure your model is in a separate 'model.py' file
import glob  # To handle multiple files
import pandas as pd

# Load the trained model
def load_model(model_path, input_dim, hidden_dim, output_dim):
    model = FeedForwardNN(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to preprocess input data
def preprocess_input(features, mean,std):
    features = features.reshape(1, -1)  # Reshape for batch processing
    features = (features-mean)/std
    features = torch.FloatTensor(features)  # Convert to tensor
    return features

# Make predictions on multiple MFCC features
def predict(feature, model, mean,std):
    feature = preprocess_input(feature, mean,std)
    with torch.no_grad():
        output = model(feature).squeeze()
        prediction = torch.sigmoid(output).item()  # Convert to probability
        label = "Male" if prediction > 0.65 else "Female"
    return label,prediction

if __name__ == "__main__":
    model_path = "best_model.pth"
    input_dim = 13 * 130  # Based on training shape
    hidden_dim = 512
    output_dim = 1
    
    # Load trained model
    model = load_model(model_path, input_dim, hidden_dim, output_dim)
    
    
    
    # Load multiple MFCC feature files
    mfcc_files = ["denoised_source_1.npy","denoised_source_2.npy"]

    
    # Load features from all files
    denoised_source_1=np.load("denoised_source_1.npy")
    denoised_source_2=np.load("denoised_source_2.npy")

    features_list = [denoised_source_1,denoised_source_2]
    
    # Predict for all files
    predictions=[]
    mean=np.load(r".\scaler_mean.npy")
    std=np.load(r".\scaler_scale.npy")
   
    for feature in features_list:
        predictions.append(predict(feature,model,mean,std))
        

    # Print results
    for file, (label, confidence) in zip(mfcc_files, predictions):
        if label=="Female":print(f"File: {file} | Predicted: {label} (Confidence: {1-confidence:.4f})")
        else:print(f"File: {file} | Predicted: {label} (Confidence: {confidence:.4f})")
