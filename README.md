#### Breast Cancer Prediction Model
This project implements a deep learning model for breast cancer classification using data from the popular `breast_cancer` dataset provided by `scikit-learn`. The goal is to classify whether a given set of input features indicates a malignant or benign tumor. The model is built using TensorFlow and Keras and achieves high accuracy in prediction.

#### Dataset Information
The breast cancer dataset from `scikit-learn` contains 30 features extracted from digitized images of breast mass, describing the characteristics of cell nuclei. It is a commonly used dataset for binary classification problems, with the labels indicating:
- **0**: Malignant
- **1**: Benign

#### Code Breakdown

1. **Loading Libraries**:
   The necessary libraries are imported, including `numpy`, `pandas`, `matplotlib`, `scikit-learn`, and `tensorflow`.

2. **Data Loading**:
   The `breast_cancer` dataset is loaded using `sklearn.datasets.load_breast_cancer()`. The data is then converted into a Pandas DataFrame for better visualization and analysis, with columns labeled according to the dataset's feature names.

3. **Data Preparation**:
   - The feature set `x` is created by dropping the 'label' column from the DataFrame, while `y` holds the target labels.
   - The dataset is split into training and test sets using an 80-20 split through `train_test_split` with a `random_state` of 2 for reproducibility.

4. **Feature Scaling**:
   Standardization is performed using `StandardScaler` to scale the features for improved model performance.

5. **Model Building**:
   - A Sequential model from `Keras` is used.
   - The architecture includes:
     - **Flatten layer**: Converts the input shape (30 features) into a 1D array.
     - **Dense layer**: A hidden layer with 20 neurons and ReLU activation function.
     - **Output layer**: A single neuron with a sigmoid activation function for binary classification.
   
6. **Model Compilation**:
   The model is compiled with the following:
   - **Optimizer**: `Adam`
   - **Loss function**: `binary_crossentropy`
   - **Metrics**: `accuracy`

7. **Model Training**:
   The model is trained using the `fit` method for 50 epochs with a 10% validation split to monitor validation performance during training.

8. **Model Evaluation**:
   The test set is evaluated using `model.evaluate()`, and the final accuracy score is printed.

#### Results
The model achieves an accuracy of **97.37%** on the test set, indicating that the model performs well in predicting whether tumors are malignant or benign.

#### Conclusion
This deep learning approach demonstrates an effective way to build a neural network model for binary classification problems such as breast cancer prediction. The project showcases essential data preprocessing steps, model training, and evaluation that can be adapted for similar classification tasks.
