# 🌾 Crop Recommendation System Using Ensemble Learning

This documentation provides a detailed explanation of the Crop Recommendation System project implemented using ensemble learning in Python. The goal of this system is to recommend the most suitable crop to grow based on environmental and soil conditions using multiple machine learning techniques.

## Overview

The project employs a dataset containing features like Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH, and rainfall, with a target label representing different crop types. The model uses an ensemble of Random Forest, Support Vector Machine (SVM), and Convolutional Neural Network (CNN) classifiers, achieving an accuracy of 99%.

## Libraries Used

The following Python libraries were used to implement the model:
- **NumPy:** For numerical operations and array manipulation
- **Pandas:** For data manipulation and analysis
- **Matplotlib & Seaborn:** For data visualization
- **TensorFlow/Keras:** For building and training the CNN model
- **Scikit-learn:** For Random Forest, SVM, data preprocessing and evaluation metrics

## Project Structure

**1. Data Loading and Preprocessing:**
- The dataset is loaded from a CSV file, and the target variable (**label**) is encoded using **LabelEncoder**
- Features are standardized using **StandardScaler**
- The dataset is split into training and test sets using **train_test_split** with a 90/10 ratio

**2. Model Architecture:**
   
The ensemble combines three different models:
- **Random Forest Classifier:** An ensemble decision tree-based algorithm
- **Support Vector Machine (SVM):** Using RBF kernel for non-linear classification
- **Convolutional Neural Network (CNN):** Sequential model with:
  - Conv1D layer with 32 filters and ReLU activation
  - MaxPooling1D layer
  - Flattening layer
  - Dense hidden layer with 128 neurons and ReLU activation
  - Output layer with softmax activation

**3. Model Training:**
- Each individual model is trained on the same training data
- The CNN model is trained for 100 epochs with a batch size of 32
- Input data is properly reshaped for CNN processing

**4. Ensemble Strategy:**
- Predictions from all three models are combined using a simple voting mechanism
- The class with the majority vote becomes the final prediction
- This ensemble approach improves robustness and accuracy

**5. Model Evaluation:**
- The ensemble model is evaluated using classification report and confusion matrix
- Accuracy is calculated and displayed with four decimal places
- Visual representation of the confusion matrix for better interpretation

**6. Crop Recommendation Function:**
- A function `recommend_crop()` is implemented to take user input (soil and environmental parameters)
- The function processes inputs through all three models and uses voting to make the final recommendation

## Evaluation Results

- **Test Accuracy:** 99%
- The ensemble model demonstrates superior performance compared to individual models
- The confusion matrix shows minimal misclassifications across all crop types
- The ensemble approach effectively handles the multiclass nature of the problem

## Usage

To get a crop recommendation, use the `recommend_crop()` function with the following parameters:
- N: Nitrogen content in soil (mg/kg)
- P: Phosphorus content in soil (mg/kg)
- K: Potassium content in soil (mg/kg)
- temperature: Temperature in degrees Celsius
- humidity: Relative humidity in percentage
- ph: pH value of the soil
- rainfall: Rainfall in mm

Example:
```python
recommended_crop = recommend_crop(90, 42, 43, 25, 86, 6.5, 220)
print("Recommended Crop:", recommended_crop)

## Conclusion
The crop recommendation system built with ensemble learning demonstrates exceptional accuracy (99%) in recommending the most suitable crop based on various environmental and soil conditions. By combining multiple machine learning techniques, the system provides robust and reliable recommendations that can help farmers make informed decisions regarding crop selection based on real-time data.
