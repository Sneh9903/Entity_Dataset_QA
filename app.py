from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import os
import random
import shutil





class FeedForwardNN(nn.Module):
    def __init__(self, input_dim):
        super(FeedForwardNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output one score
        )

    def forward(self, x):
        return self.fc(x)

# Define the main model that combines ViT and the Feed-Forward NN
class ImageQualityAssessmentModel(nn.Module):
    def __init__(self, transformer_model, input_dim):
        super(ImageQualityAssessmentModel, self).__init__()
        self.transformer_model = transformer_model
        self.feed_forward_nn = FeedForwardNN(input_dim)

    def forward(self, images):
        batch_size, num_images, _, _, _ = images.shape
        embeddings = []

        for i in range(num_images):
            output = self.transformer_model(images[:, i].to(device))  # Pass each image individually
            embeddings.append(output.last_hidden_state.mean(dim=1))  # 768-D vector

        concatenated_embeddings = torch.cat(embeddings, dim=1)  # Concatenate to [batch_size, 768*num_images]
        return self.feed_forward_nn(concatenated_embeddings)

# Initialize the components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
transformer_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)

# Load the trained model
model = ImageQualityAssessmentModel(transformer_model, input_dim=768*16).to(device)
model.load_state_dict(torch.load("image_quality_assessment_model.pth", map_location=device))
model.eval()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    folder = request.files.getlist('folder')
    n = int(request.form['n'])
    
    # Save the uploaded folder
    folder_path = os.path.join("uploads", "uploaded_images")
    os.makedirs(folder_path, exist_ok=True)
    for file in folder:
        file_path = os.path.join(folder_path, os.path.basename(file.filename))
        file.save(file_path)
        print(f"Saved file: {file_path}")  # Debugging: Print each saved file path
    
    # Get all image file names in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Debugging: Print the number of files and file paths
    print(f"Number of files saved: {len(image_files)}")
    print(f"File paths: {image_files}")
    
    # Check if there are at least 16 images
    if len(image_files) < 16:
        shutil.rmtree(folder_path)
        return render_template('index.html', prediction_text='Error: Not enough images. Please upload at least 16 images.')

    # Randomly select 16 images n times
    image_sets = []
    for _ in range(n):
        selected_images = random.sample(image_files, 16)
        image_sets.append(selected_images)
    
    # Create a dataset of these image sets
    class ImageSetDataset(Dataset):
        def __init__(self, image_sets, feature_extractor, image_folder):
            self.image_sets = image_sets
            self.feature_extractor = feature_extractor
            self.image_folder = image_folder

        def __len__(self):
            return len(self.image_sets)

        def __getitem__(self, idx):
            image_names = self.image_sets[idx]
            images = [Image.open(os.path.join(self.image_folder, img_name)).convert("RGB") for img_name in image_names]
            inputs = self.feature_extractor(images, return_tensors="pt")
            return inputs

    dataset = ImageSetDataset(image_sets, feature_extractor, folder_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Pass these image sets through the model to get predictions
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['pixel_values'].squeeze(1).to(device)
            outputs = model(inputs)
            predictions.append(outputs.item())

    # Average the predictions
    avg_prediction = sum(predictions) / len(predictions)

    # Clean up the uploaded folder
    shutil.rmtree(folder_path)

    return render_template('index.html', prediction_text='Average Prediction: {:.2f}'.format(avg_prediction))

if __name__ == "__main__":
    app.run(debug=True)