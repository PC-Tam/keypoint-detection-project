# Glaucoma Classification via Traditional Computer Vision & Machine Learning

This project implements a pipeline to classify fundus images into Normal or Glaucoma using traditional computer vision techniques (Harris Corner, FAST, ORB) and Machine Learning models (SVM, Random Forest, KNN, Naive Bayes), without relying on Deep Learning.

## Features
- **Traditional Feature Extraction:** Uses Harris Corner Detection, FAST, and ORB to identify keypoints.
- **Bag of Visual Words (BoVW):** Converts local ORB descriptors into global image features using K-Means clustering.
- **Machine Learning Models:** Trains SVM, Random Forest, KNN, and Gaussian Naive Bayes on the extracted features.
- **Interactive UI:** Streamlit application for end-to-end inference and visualization.

## Setup & Installation

1. Make sure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

This project is built to work with the **ACRIMA** dataset, or optionally **RIM-ONE DL**.

### Automatic Download
If you have your Kaggle API token set up (`~/.kaggle/kaggle.json`), the pipeline will attempt to automatically download the ACRIMA dataset when you run `run_pipeline.py`.

### Manual Download
If the automatic download fails, follow these steps:
1. Download the ACRIMA dataset from [Kaggle](https://www.kaggle.com/datasets/mloey1/acrima).
2. Extract the dataset.
3. Place all the `.jpg` / `.png` fundus images directly into `data/raw/ACRIMA/images/`.

*Note on labels:* By default, the ACRIMA dataset names glaucoma images with `_g_` in the filename and normal images without `_g_`. The pipeline uses this naming convention to assign labels automatically.

## Running the Pipeline

To run the entire pipeline (dataset check, preprocessing, feature extraction, ML training, evaluation, and saving results):

```bash
python src/run_pipeline.py
```

The pipeline will:
- Check for the dataset and extract metadata.
- Preprocess images (Green Channel Extraction, CLAHE).
- Extract Harris, FAST, and ORB features (with BoVW).
- Train SVM, RF, KNN, and GNB models on all feature types.
- Evaluate the models and save the metrics, reports, and confusion matrices to the `outputs/` folder.

## Interpreting Results

After the pipeline finishes, check the `outputs/` directory:
- **`outputs/reports/results.csv`**: Contains Accuracy, Precision, Recall, F1-Score, and Specificity for each feature-model combination.
- **`outputs/reports/classification_report.txt`**: Detailed classification metrics.
- **`outputs/figures/`**: Contains confusion matrix plots and sample keypoint visualizations.

*Expected Performance:* For traditional CV methods without deep learning, an Accuracy or F1-Score above 70% is considered acceptable, and above 80% is considered good. Deep learning typically performs better, but these baseline models help understand the distinct visual features of Glaucoma.

## Explaining the Features
- **Harris Corners:** Detects corners in the image. We aggregate the number of corners, response statistics (mean, max, std), and their spatial distribution.
- **FAST Keypoints:** A faster corner detection algorithm. Similar to Harris, we build a fixed-size vector summarizing the keypoints.
- **ORB + BoVW:** ORB extracts scale and rotation-invariant keypoints and descriptors. BoVW clusters these descriptors into "visual words" to create a histogram representing the image, providing a richer textural description than simple corner statistics.

## Running the Demo UI

You can run an interactive web application to upload images and see the features and predictions in real-time:

```bash
streamlit run app.py
```
