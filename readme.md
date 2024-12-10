<h1>Real Estate Price Prediction</h1>

This project predicts real estate prices using a linear regression model. It combines several datasets, cleans and preprocesses the data, and trains a machine learning model to provide accurate predictions. The project is modular and easy to extend, ensuring maintainability and scalability.

## **Project Structure**
├── data/ # Folder for raw and processed datasets 
<br>├── graphs/ # Folder for generated plots 
<br>├── main.py # Main script to execute the pipeline 
<br>├── cleaning_datasets.py # Module for data cleaning and merging 
<br>├── cleaning_feature_engineering.py # Module for feature engineering 
<br>├── linear_regression_model.py # Module for training and evaluating the model 
<br>├── requirements.txt # List of dependencies 
<br>└── README.md # Project documentation


## **How to Use**

### **1. Setup**
1. Clone the repository:<br>
   git clone https://github.com/your-repo-name/real-estate-price-prediction.git<br>
   cd real-estate-price-prediction<br>
Install the required dependencies:<br>
pip install -r requirements.txt<br>
Ensure the raw datasets are in the data/ directory.

2. Run the Pipeline<br>
Execute the pipeline by running:
python main.py<br>
This script performs the following steps:<br>
Merges and cleans the datasets.<br>
Preprocesses the data (handles outliers, missing values, and categorical variables).<br>
Trains a linear regression model.<br>
Saves cleaned data, metrics, and visualizations.

3. Outputs<br>
Processed Dataset: data/dataset-preprocessed.csv<br>
Model Evaluation: Metrics printed in the console.<br>
Visualization: A scatter plot comparing predicted and real values saved in the graphs/ folder.

4. Example Dataset<br>
Ensure your raw datasets have columns like:<br>
Price, Property type, Living area, Building condition, Zip code, and more.<br>
Refer to the project scripts for detailed dataset expectations.

### **2. Modules Overview** 
1. cleaning_datasets.py<br>
Handles merging and cleaning of multiple datasets, renames columns, and ensures consistent formatting.

2. cleaning_feature_engineering.py<br>
Includes functions for handling outliers, replacing missing values, and transforming categorical variables into numerical data.

3. linear_regression_model.py<br>
Implements a linear regression model, evaluates it using metrics (MAE, RMSE, MAPE, R²), and generates visualizations.

4. data-analysis<br>
Graphs and analysis that helps in choosing the columns that will be used in the linear regression.

Dependencies :
Python 3.8+,
pandas,
numpy,
scikit-learn,
matplotlib <br>
For detailed dependencies, see requirements.txt.
