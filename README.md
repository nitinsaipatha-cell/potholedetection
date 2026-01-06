# Pothole Detection Using Deep Learning (ResNet)

This repository contains a real-time pothole detection system implemented using deep learning. The model is trained using a ResNet-18 architecture and performs pothole detection using a live webcam feed.
The system classifies each video frame as either pothole or no_pothole and displays the prediction directly on the video stream.

## Project Overview

This project uses a trained ResNet-18 model to detect potholes in real time. Video frames captured from a webcam are processed frame by frame, passed through the trained model, and the predicted class label is displayed on the screen.

The model automatically runs on GPU if available; otherwise, it runs on CPU.

## Model Details

- Architecture: ResNet-18  
- Framework: PyTorch  
- Classes:
  - no_pothole
  - pothole  
- Input Size: 224 × 224 RGB images  
- Model File: pothole_model.pth  

## Repository Structure

|-- pothole_model.pth        # Trained model weights  
|-- video prediction.py      # Real-time webcam detection script  
|-- README.md                # Project documentation  

## Requirements

- Python 3.x  
- PyTorch  
- Torchvision  
- OpenCV  
- Pillow  

Install dependencies using:
pip install torch torchvision opencv-python pillow

## How to Run

1. Clone the repository:
git clone https://github.com/nitinsaipatha-cell/potholedetection.git
cd pothole-detection  

3. Ensure pothole_model.pth and video prediction.py are in the same directory.

4. Run the script:
python "video prediction.py"

5. The webcam will open and display predictions:
- Green text: no_pothole  
- Red text: pothole  

5. Press Q to exit.

## Device Support

- Uses CUDA-enabled GPU if available  
- Falls back to CPU if GPU is not available  

## Applications

- Road condition monitoring  
- Smart city infrastructure  
- Real-time pothole detection  
- Academic and research projects  

## Disclaimer

This project is developed for educational and academic purposes only. Detection accuracy may vary depending on lighting conditions, camera quality, and road surface variations.


## Author

Nitin Sai Patha
