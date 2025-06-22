# Plant Health Detector

## Overview
The **Plant Health Detector** is a machine learning-based web application that identifies diseases in tomato and potato leaves using a Convolutional Neural Network (CNN). Built with TensorFlow and deployed via Streamlit, it allows users to upload leaf images and receive disease predictions with a user-friendly interface. This project showcases expertise in deep learning, image processing, and web application development, as part of my portfolio at [DEVBALIYAN21](https://github.com/DEVBALIYAN21).

## Features
- **Disease Detection**: Classifies diseases in tomato and potato leaves, including Early Blight, Late Blight, Bacterial Spot, and more, with a "healthy" class for unaffected leaves.
- **CNN Model**: Utilizes a TensorFlow/Keras-based CNN with multiple convolutional and pooling layers for accurate image classification.
- **Data Augmentation**: Applies transformations (rotation, zoom, flip) to enhance model robustness using `ImageDataGenerator`.
- **Streamlit Interface**: Provides an intuitive web interface for uploading multiple images (minimum 3) and displaying predictions with visual feedback.
- **Model Training**: Includes training and fine-tuning scripts with early stopping and model checkpointing to save the best-performing model.
- **Visualization**: Plots training/validation accuracy and loss to evaluate model performance.

## Technologies Used
- **Languages**: Python
- **Libraries**: TensorFlow, Keras, Streamlit, NumPy, Matplotlib, Pyngrok
- **Tools**: Google Colab, Google Drive, Ngrok
- **Concepts**: Deep Learning, Convolutional Neural Networks, Image Processing, Web Application Deployment

## Prerequisites
- Google Colab account with access to a **T4 GPU** runtime (recommended for faster training).
- Google Drive account to store the dataset and model files.
- Ngrok account with an authentication token for Streamlit deployment.
- Dataset: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) (or equivalent).

## Setup Instructions
1. **Open in Google Colab**:
   - Copy the provided notebook code into a new Google Colab notebook.
   - **Select T4 GPU Runtime**:
     - Go to `Runtime` > `Change runtime type` in Colab.
     - Choose `T4 GPU` under Hardware Accelerator to ensure efficient model training.
     - Save the runtime settings.

2. **Mount Google Drive**:
   - Run the cell to mount Google Drive: `drive.mount('/content/drive')`.
   - Authenticate using your Google account and copy the authorization code.

3. **Prepare the Dataset**:
   - Upload the `New Plant Diseases Dataset.zip` to your Google Drive.
   - Update the `zip_file_path` in the code to point to the dataset location (e.g., `/content/drive/My Drive/New Plant Diseases Dataset.zip`).
   - Ensure the dataset is structured with `train` and `valid` folders containing subfolders for each class (e.g., `Potato___Early_blight`, `Tomato___healthy`).

4. **Install Dependencies**:
   - Run the cells to import libraries (`tensorflow`, `streamlit`, `pyngrok`, etc.).
   - Install `pyngrok` if not already installed: `!pip install pyngrok`.

5. **Set Up Ngrok**:
   - Sign up for a free Ngrok account and obtain an authentication token.
   - Update the Ngrok token in the code: `ngrok.set_auth_token('YOUR_NGROK_AUTH_TOKEN')`.
  

6. **Run the Notebook**:
   - Execute the cells sequentially to:
     - Unzip the dataset.
     - Train the CNN model (10 epochs initially, then fine-tune for up to 50 epochs with early stopping).
     - Save the model to Google Drive (`Final_Updated_Plant_Disease_Model_latest.h5`).
     - Generate accuracy/loss plots.
     - Count dataset classes.
     - Test predictions with sample images.
     - Deploy the Streamlit app.

7. **Deploy the Streamlit App**:
   - Run the final cell to create and run `app.py` with Streamlit.
   - Use the Ngrok public URL (e.g., `http://<random>.ngrok.io`) to access the app.
   - Upload at least 3 images (JPG/PNG) of tomato or potato leaves to get predictions.

## Dataset Structure
The dataset should be organized as follows:
```
/content/New_Plant_Diseases_Dataset/
├── train/
│   ├── Potato___Early_blight/
│   ├── Potato___Late_blight/
│   ├── Potato___healthy/
│   ├── Tomato___Bacterial_spot/
│   ├── ...
└── valid/
    ├── Potato___Early_blight/
    ├── Potato___Late_blight/
    ├── ...
```

## Usage
1. **Training the Model**:
   - Run the training cells to train the CNN on the dataset.
   - The model is saved to Google Drive after training (`Final_Updated_Plant_Disease_Model_latest.h5`).
   - Monitor accuracy/loss plots to evaluate performance.

2. **Testing Predictions**:
   - Update `test_image_paths` with paths to your test images (e.g., `/content/2.JPG`).
   - Run the prediction cell to classify images as tomato/potato leaves with disease or healthy status.

3. **Streamlit App**:
   - Access the app via the Ngrok URL.
   - Upload 3 or more images to receive predictions.
   - The app displays the uploaded image, plant type (tomato/potato), and disease status (healthy or specific disease).

## Troubleshooting
- **T4 GPU Not Available**: Ensure you have a Colab Pro subscription or free GPU availability. If unavailable, the CPU runtime will work but be slower.
- **FileNotFoundError**: Verify the dataset and model paths in Google Drive. Ensure the zip file and extracted folders are accessible.
- **Ngrok Errors**: Check your Ngrok token and ensure port `8501` is used correctly. Restart the cell if the tunnel fails.
- **Class Mismatch**: Ensure the number of classes in `train` and `valid` directories matches the model’s output shape.

## Future Enhancements
- Add support for more plant types beyond tomatoes and potatoes.
- Integrate additional preprocessing techniques (e.g., contrast enhancement) for better accuracy.
- Deploy the Streamlit app on a dedicated platform like Streamlit Cloud for persistent access.
- Add a confidence score display for predictions in the Streamlit app.

## Repository
- **GitHub**: [Plant Health Detector](https://github.com/DEVBALIYAN21/Plant-Health-Detector) *(Create repo if not available)*
- **Status**: Actively maintained

## Demo
- *Planned*: Deploy on Streamlit Cloud for a live demo.
- *Temporary Access*: Use the Ngrok URL generated during Colab execution.

## Notes
- **T4 GPU Requirement**: The T4 GPU runtime is critical for efficient CNN training due to the computational demands of image processing. Using a CPU runtime may significantly slow down training.
- **Dataset**: Ensure the dataset is correctly structured and accessible in Google Drive to avoid errors.
- **Streamlit Deployment**: Ngrok provides temporary URLs; consider Streamlit Cloud for a permanent deployment.
- **Portfolio Integration**: This project is part of my portfolio, showcasing machine learning and web development skills. See other projects like QuizSystem and Resume Analyzer at [DEVBALIYAN21](https://github.com/DEVBALIYAN21).

## Contact
- **Email**: devbaliyan202@gmail.com
- **GitHub**: [DEVBALIYAN21](https://github.com/DEVBALIYAN21)
