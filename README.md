# [IBM Machine Learning Professional Certificate Projects](https://www.coursera.org/professional-certificates/ibm-machine-learning)

## Overview
This repository contains a collection of projects completed as part of the IBM Machine Learning Professional Certificate on Coursera. These projects span various domains of machine learning, including regression, classification, clustering, deep learning, and time series analysis. The goal was to apply machine learning techniques to real-world datasets and draw meaningful insights and predictions.

## Projects

### 1. [Supervised Learning - Regression](https://github.com/TensorSpd/IBM-Machine-Learning-Certificate-Projects/tree/main/Supervised%20Learning%20-%20Regression) 
- **Dataset**: The dataset, sourced from Kaggle, focuses on the global economic impact of the COVID-19 pandemic, with an emphasis on poverty alleviation and economic growth.
- **Objective**: To build a regression model that predicts the GDP of countries based on factors such as the Human Development Index and total COVID-19 deaths.
- **Outcome**: After applying various linear regression algorithms, the optimal model was selected based on performance, and the model can predict the GDP impact accurately.

### 2. [Supervised Learning - Classification](https://github.com/TensorSpd/IBM-Machine-Learning-Certificate-Projects/tree/main/Supervised%20Learning%20-%20Classification)
- **Dataset**: Stellar classification dataset from Kaggle, collected by the Sloan Digital Sky Survey (SDSS), consisting of 100,000 observations with 17 features describing stars.
- **Objective**: To classify celestial objects (stars, galaxies, quasars) using supervised learning techniques based on their spectral characteristics.
- **Outcome**: A classification model was trained and successfully differentiated between stars, galaxies, and quasars with high accuracy.

### 3. [Unsupervised Learning - Clustering](https://github.com/TensorSpd/IBM-Machine-Learning-Certificate-Projects/tree/main/Unsupervised%20Learning)
- **Dataset**: The Wholesale Customer dataset from UCI, which contains yearly spending in different product categories.
- **Objective**: To segment wholesale customers into clusters based on their spending behavior using clustering algorithms like K-Means and Agglomerative Clustering.
- **Outcome**: The customers were grouped into clusters that reveal purchasing patterns, and the results were visualized using cluster plots.

### 4. [Deep Learning - Image Classification with CNN](https://github.com/TensorSpd/IBM-Machine-Learning-Certificate-Projects/tree/main/Deep%20Learning)
- **Dataset**: General image dataset sourced from Kaggle and supplemented by fetching images using `gdown` from external URLs.
- **Objective**: To train a Convolutional Neural Network (CNN) model for classifying images and predicting similar images from a dataset of dress categories linked to Amazon URLs.
- **Outcome**: The CNN model successfully identified and classified images, and feature engineering was used to enhance predictions of similar images from the dataset.

### 5. [Time Series Analysis](https://github.com/TensorSpd/IBM-Machine-Learning-Certificate-Projects/tree/main/Time%20Series)
- **Dataset**: Stock price data for Apple Inc., collected from Yahoo Finance, spanning from 1980 to 2021.
- **Objective**: To apply deep learning techniques such as Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) to forecast future stock prices of Apple Inc.
- **Outcome**: LSTM outperformed RNN in reducing prediction error, particularly due to its ability to handle long-term dependencies and avoid the vanishing gradient problem.

## Key Learnings
These projects helped reinforce key machine learning concepts such as regression, classification, clustering, and deep learning. Through hands-on experience with real-world datasets, I gained proficiency in Python programming, data preprocessing, model building, and evaluation techniques. The experience also deepened my understanding of advanced machine learning models like CNN and LSTM for image classification and time series forecasting.

## Tools and Technologies Used
- **Languages**: Python
- **Libraries**: Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn, Keras, TensorFlow
- **Machine Learning Techniques**: Regression, Classification, Clustering, CNN, RNN, LSTM
- **Other Tools**: Jupyter Notebooks, Git, Kaggle Datasets, gdown
