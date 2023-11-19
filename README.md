# aml-bank-marketing-campaing
Project of Advanced Machine Learning (AML) Course for the Master in Data Science Program of Universitat Politècnica de Catalunya (UPC)

## Instructions for Executing Analysis Notebooks

* Navigate to the `notebooks` folder included in the repository.
* Make sure `bank-full.csv` file is located under the `data` directory.
* Execute the notebooks with the following order:
	* `01.EDA-feature-extraction.ipynb`
		* This notebook generates in the end the updated dataset (named `bank_marketing_new.csv`, under `notebooks` directory), which includes all the changes introduced during **EDA**.
		* The new file (`bank_marketing_new.csv`) is being used from the second notebook in order to complete the modeling part.
	* `02.Experiments.ipynb`
    * `03.Modelling.ipynb`
        * In this notebook we have the main model executions, which we include in the results table of the report. It contains a data loading and several hyperparametrization executions on full and undersampled training set. 
* All notebooks generate some extra **csv** files, containing information used during the analysis
	* E.g.: `chi-2.csv`, `accuracy_results.csv`, and `recall_results.csv` files mentioned in the report are generated during the execution of the above-mentioned notebooks. They can also be found in this repository.
* Uncomment the first cell of the notebooks in order to install missing libraries.
* Click on the `Run all` button of the notebooks to reproduce the results of the whole project.

## Analysis Includes
1. Exploratory Data Analysis (EDA)
    * Univariate Exploratory Analysis
        * Univariate Outliers Analysis
    * Bivariate Exploratory Analysis
        * Multivariate Outlier Analysis
    * Feature Extraction
2. Experiments
    * Preprocessing
    * Dataset Splits
    * Learning Algorithms
    * Model Comparison & Hyper Parameter Tuning
    	* 5-fold Cross Validation on the following models, with multiple hyper-parameter values:
     		* Logistic Regression
       		* Random Forest
       		* SVM (with `linear`, `sigmoid`, `rbf` and `poly` kernels)
       		* Gradient Boosting
       		* kNN
       		* Decision Tree
       		* Naive Bayes
    * Final Model Performance Analysis (Generalization, bias, variance analysis)
        * Final model is an `SVM with signoid kernel`
        * Training-Test Error Analysis on the `C` Parameter
        * Training-Test Error Analysis on the `gamma` Parameter
        * Training-Test Error Analysis on the `class_weight` Parameter
        * Training-Test Error Analysis on the `Training Data Size`
        * Final Performance Metrics
        * Interpretability of the Final Model
3. Limitations & Future Work

## Executed Notebooks
The following two `html` files, present the notebooks of the project executed.
* `01.EDA-feature-extraction.html`
* `02.Experiments.html`
* `03.Modelling.ipynb`
