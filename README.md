README for Multi-Class Classification Models (Google Colab) - HumanActivityRecognitionModels

This README file provides an overview of the code and usage instructions for a Python script that demonstrates multi-class classification using various machine learning models in a Google Colab environment. The script covers the following models:
- Logistic Regression
- LightGBM (Light Gradient Boosting Machine)
- XGBoost
- Decision Tree
- Random Forest

Prerequisites:
Before running the script in Google Colab, you need to ensure that you have a Google Colab environment set up. Additionally, you should have a dataset named data.csv uploaded to your Colab environment. You can upload the dataset directly from your local machine or from cloud storage services.

Usage:
- Open a new or existing Google Colab notebook.
- Upload the script to your Colab environment.
- Make sure that you have the required Python packages installed within your Colab environment.
- You can install these packages using the following commands within a Colab cell:
```
!pip install numpy pandas scikit-learn matplotlib xgboost lightgbm
```
- Upload the data.csv dataset to your Colab environment. 

Output:
The script provides output for each machine learning model, including accuracy, confusion matrices, classification reports, and cross-validation scores. Additionally, ROC curves are visualized for each class for each model, highlighting the model's performance for multi-class classification.

Note:
- LightGBM can be sensitive to the choice of parameters. You can experiment with different hyperparameters as needed.
- For the Random Forest model, hyperparameter tuning is demonstrated using GridSearchCV. The best parameters and the corresponding accuracy are displayed.
- Make sure to have the data.csv dataset uploaded to your Colab environment.

Thank you for using this multi-class classification demonstration script in Google Colab. If you have any questions or need assistance, please feel free to reach out.
