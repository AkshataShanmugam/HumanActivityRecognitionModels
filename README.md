# Multi-Class Classification Models - Human Activity Recognition Model

This README file provides an overview of the code and usage instructions for a Python script that demonstrates multi-class classification using various machine learning models in a Google Colab environment. The script covers the following models:
- Logistic Regression
- LightGBM (Light Gradient Boosting Machine)
- XGBoost
- Decision Tree
- Random Forest

## WISDM Dataset

- **Dataset URL:** [WISDM Dataset](https://www.cis.fordham.edu/wisdm/dataset.php)
- **Last Updated:** Dec. 2, 2012

The WISDM dataset contains data collected under controlled laboratory conditions. The dataset statistics are as follows:

- **Raw Time Series Data**
  - Number of examples: 1,098,207
  - Number of attributes: 6
  - Missing attribute values: None

- **Class Distribution**
  - Walking: 424,400 (38.6%)
  - Jogging: 342,177 (31.2%)
  - Upstairs: 122,869 (11.2%)
  - Downstairs: 100,427 (9.1%)
  - Sitting: 59,939 (5.5%)
  - Standing: 48,395 (4.4%)

## Dataset Preprocessing

Before using the dataset in the multi-class classification models, the file `WISDM_ar_v1.1_raw_about.txt` was converted to a CSV format using the following code:

```python
import pandas as pd

# Read the data from the text file
with open("data.txt", "r") as txt_file:
    data_lines = txt_file.readlines()

# Process the data and convert it to a list of dictionaries
data_list = []
for line in data_lines:
    parts = line.strip().split(',')

    try:
        user = int(parts[0])
        activity = parts[1]
        timestamp = int(parts[2])
        x_acceleration = float(parts[3])
        y_acceleration = float(parts[4])
        z_acceleration = float(parts[5].rstrip(';'))  # Remove semicolon

        data_list.append({
            'user': user,
            'activity': activity,
            'timestamp': timestamp,
            'x-acceleration': x_acceleration,
            'y-acceleration': y_acceleration,
            'z-acceleration': z_acceleration
        })
    except ValueError:
        print(f"Skipping line: {line}")

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(data_list)

# Save the DataFrame to a CSV file
df.to_csv("data.csv", index=False)
```

The resulting `data.csv` file is used for the multi-class classification models in this repository.

## Prerequisites:
Before running the script in Google Colab, you need to ensure that you have a Google Colab environment set up. Additionally, you should have a dataset named `data.csv` uploaded to your Colab environment. You can upload the dataset directly from your local machine or from cloud storage services.

## Usage:
- Open a new or existing Google Colab notebook.
- Upload the script to your Colab environment.
- Make sure that you have the required Python packages installed within your Colab environment.
- You can install these packages using the following commands within a Colab cell:
```python
!pip install numpy pandas scikit-learn matplotlib xgboost lightgbm
```
- Upload the `data.csv` dataset to your Colab environment. 

## Output:
The script provides output for each machine learning model, including accuracy, confusion matrices, classification reports, and cross-validation scores. Additionally, ROC curves are visualized for each class for each model, highlighting the model's performance for multi-class classification.

## Accuracy Analysis:
- Logistic Regression: Achieved an accuracy of 0.49, indicating limited performance.
- LightGBM: Accuracy of 0.94.
- XGBoost: Performed well with an accuracy of 0.95.
- Decision Tree: Accuracy of 0.99.
- Random Forest: Achieved an accuracy of 0.99, with the best model's parameters provided.

### Note:
- LightGBM can be sensitive to the choice of parameters. You can experiment with different hyperparameters as needed.
- For the Random Forest model, hyperparameter tuning is demonstrated using GridSearchCV. The best parameters and the corresponding accuracy are displayed.
- Make sure to have the `data.csv` dataset uploaded to your Colab environment.

Thank you for using this multi-class classification demonstration script in Google Colab. If you have any questions or need assistance, please feel free to reach out.
