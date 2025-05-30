# VPN Detection Models - Cybersecurity Competition

This project implements and compares multiple machine learning models for VPN vs Non-VPN traffic classification. The primary goal is to explore, visualize the dataset, train various classification models, compare their accuracy and training speed, and save the best-performing models for potential deployment.

**Author:** Mahdi Ben Ameur
**Task:** VPN Traffic Classification

---

## Objectives

* Explore and visualize the traffic dataset.
* Train multiple machine learning classification models.
* Compare the accuracy and training speed of the trained models.
* Save the trained models for future use or deployment.

---

## Project Structure

.
├── backup/
│   └── ... (backup files)
├── data/
│   ├── Scenario A1-ARFF/
│   │   └── ... (ARFF data files)
│   ├── Scenario A2-ARFF/
│   │   └── ... (ARFF data files)
│   └── Scenario B-ARFF/
│       └── ... (ARFF data files)
├── fursah_venv/
│   └── ... (Python virtual environment files - typically in .gitignore)
├── saved_models/
│   ├── decision_tree_model.joblib
│   ├── lightgbm_model.joblib
│   ├── logistic_regression_model.joblib
│   ├── model_comparison_results.csv
│   ├── model_metadata.joblib
│   ├── random_forest_model.joblib
│   └── xgboost_model.joblib
├── .gitignore
├── convert_to_csv.py
├── data.csv
├── Fursah.ipynb
└── merge.py


---

## Dataset

The primary dataset used for training and evaluation is `data.csv`. This CSV file is derived from initial ARFF files located in the `data/` subdirectories (`Scenario A1-ARFF`, `Scenario A2-ARFF`, `Scenario B-ARFF`). The `convert_to_csv.py` script might be used for this conversion or for merging different data sources.

The dataset contains various network flow features, and the target variable `class1` indicates whether the traffic is "VPN" or "Non-VPN". This is later converted to a binary flag `vpn_flag` (1 for VPN, 0 for Non-VPN) for modeling.

---

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv fursah_venv
    source fursah_venv/bin/activate  # On Windows use `fursah_venv\Scripts\activate`
    ```

3.  **Install required libraries:**
    The project uses several Python libraries for data manipulation, machine learning, and visualization. You can install them using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm joblib
    ```
    *(Note: `catboost` is commented out in the notebook but can be added if needed.)*

---

## Usage

The main workflow for model training and evaluation is contained within the Jupyter notebook: `Fursah.ipynb`.

1.  Ensure all dependencies are installed (see Setup and Installation).
2.  Launch Jupyter Notebook or Jupyter Lab:
    ```bash
    jupyter notebook
    ```
    or
    ```bash
    jupyter lab
    ```
3.  Open `Fursah.ipynb` and run the cells sequentially.

The notebook covers the following steps:
* **Data Loading and Initial Exploration:** Loads `data.csv` and provides an initial overview.
* **Data Preprocessing:**
    * Creates a binary target variable `vpn_flag`.
    * Cleans the data by handling missing values (imputing with median) and problematic entries.
* **Exploratory Data Analysis (EDA):** Includes statistical summaries. (Further EDA visualizations might be present in the notebook).
* **Feature Scaling and Data Splitting:** Prepares data for model training.
* **Model Training and Evaluation:** Trains and evaluates several classification models, including:
    * Logistic Regression
    * Decision Tree
    * Random Forest
    * XGBoost
    * LightGBM
    * (Other models like Gradient Boosting, SVM, KNN, GaussianNB might be experimented with as per imports).
* **Model Comparison:** Compares models based on accuracy and training time. Results might be saved in `saved_models/model_comparison_results.csv`.
* **Saving Models:** Trained models are saved as `.joblib` files in the `saved_models/` directory.

---

## Key Files and Directories

* **`Fursah.ipynb`**: The main Jupyter Notebook containing the entire analysis, model training, and evaluation pipeline.
* **`data.csv`**: The primary dataset used for the analysis.
* **`convert_to_csv.py`**: A Python script likely used for converting or preprocessing ARFF files into the `data.csv` format or merging datasets.
* **`merge.py`**: A Python script, potentially for merging datasets or results.
* **`data/`**: Directory containing the raw data in ARFF format, categorized into scenarios.
* **`saved_models/`**: Directory where trained machine learning models (`.joblib` files) and model comparison results are stored.
* **`.gitignore`**: Specifies intentionally untracked files that Git should ignore (e.g., `fursah_venv/`, `backup/`).
* **`backup/`**: Directory likely containing backups of important files or previous versions.

---

## Models Implemented

The notebook explores and evaluates the following machine learning models for VPN traffic classification:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* XGBoost Classifier
* LightGBM Classifier

The performance (accuracy, training time, etc.) of these models is compared to identify the most suitable ones for the task.
