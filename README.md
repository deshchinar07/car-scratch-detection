# Car Scratch Detection

## About
This project provides an automated system to detect scratches and damage on cars using computer vision and image processing techniques in Python. It aims to assist in vehicle damage assessment for insurance, repair estimation, and quality checks.

## Features
- Detection of scratches from both single and multiple car images.
- Wheel detection capabilities to isolate relevant regions of interest.
- Advanced image processing to highlight damages and scratches.
- Easy-to-use scripts for batch processing of images.

## Code Files Overview

### wheel_detection.py
This script detects wheels in car images using image processing methods. It helps to focus damage detection on critical areas near wheels.

### multipleImg_scratch_detection.py
Handles scratch detection across multiple images. It processes batches of images to identify and visualize areas with potential scratches, consolidating results for overall assessment.

### scratch_detection.py
Core scratch detection logic implemented here. Uses edge detection, contour finding, and morphological image operations to highlight scratches on car surfaces in a given image.

## Technologies
- Python 3.x
- OpenCV for image processing
- NumPy for numerical operations
- Matplotlib for visualizing detections

## How to Use

### Installation
1. Clone the repository
git clone https://github.com/deshchinar07/car-scratch-detection.git

2. Install required dependencies (preferably in a virtual environment)
pip install -r requirements.txt

(Ensure required libraries like OpenCV, NumPy, and Matplotlib are installed.)

### Running the Scripts
- To detect wheels in images:
python wheel_detection.py --input path_to_image.jpg

- To detect scratches in a single image:
python scratch_detection.py --input path_to_image.jpg

- To run scratch detection on multiple images:
python multipleImg_scratch_detection.py --input_folder path_to_images_folder

text

## Contribution
Contributions are welcome! Feel free to open issues or submit pull requests to improve functionality.

## Contact
For support or questions, open an issue on GitHub or reach out to the maintainer.
