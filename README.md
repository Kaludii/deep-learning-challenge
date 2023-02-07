<div id="bootcamp"><img style="display: none;" src="https://static.bc-edx.com/data/dl-1-1/m21/lms/img/banner.jpg" alt="lesson banner" />

### Background

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

* **EIN** and **NAME**&mdash;Identification columns
* **APPLICATION_TYPE**&mdash;Alphabet Soup application type
* **AFFILIATION**&mdash;Affiliated sector of industry
* **CLASSIFICATION**&mdash;Government organization classification
* **USE_CASE**&mdash;Use case for funding
* **ORGANIZATION**&mdash;Organization type
* **STATUS**&mdash;Active status
* **INCOME_AMT**&mdash;Income classification
* **SPECIAL_CONSIDERATIONS**&mdash;Special considerations for application
* **ASK_AMT**&mdash;Funding amount requested
* **IS_SUCCESSFUL**&mdash;Was the money used effectively

### Instructions

### Step 1: Preprocessed the Data

Using my knowledge of Pandas and scikit-learn’s `StandardScaler()`, I preprocessed the dataset. This step prepared me for Step 2, where you'll compile, train, and evaluate the neural network model.

1. Read in the `charity_data.csv` to a Pandas DataFrame, and be sure to identify the following in your dataset:
  * What variable(s) are the target(s) for your model?
  * What variable(s) are the feature(s) for your model?

2. Dropped the `EIN` and `NAME` columns.

3. Determined the number of unique values for each column.

4. For columns that have more than 10 unique values, determined the number of data points for each unique value.

5. Used the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then checked if the binning was successful.

6. Used `pd.get_dummies()` to encode categorical variables.

7. Splited the preprocessed data into a features array, `X`, and a target array, `y`. Used these arrays and the `train_test_split` function to split the data into training and testing datasets.

8. Scaled the training and testing features datasets by creating a `StandardScaler` instance, fitting it to the training data, then using the `transform` function.

### Step 2: Compiled, Trained, and Evaluated the Model

Used ymy knowledge of TensorFlow, I designed a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset.

1. Continued using the Jupyter Notebook in which you performed the preprocessing steps from Step 1.

2. Created a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

3. Created the first hidden layer and choose an appropriate activation function.

4. Added a second hidden layer with an appropriate activation function.

5. Created an output layer with an appropriate activation function.

6. Checked the structure of the model.

7. Compiled and trained the model.

8. Created a callback that saves the model's weights every five epochs.

9. Evaluated the model using the test data to determine the loss and accuracy.

10. Saved and exported the results to an HDF5 file. Name the file `charity_model.h5`.

### Step 3: Optimized the Model

Used my knowledge of TensorFlow to optimize the model to achieve a target predictive accuracy higher than 75%.

Used any or all of the following methods to optimize your model:

* Adjusted the input data to ensure that no variables or outliers are causing confusion in the model, such as:
  * Dropped more or fewer columns.
  * Created more bins for rare occurrences in columns.
  * Increased or decreased the number of values for each bin.
* Added more neurons to a hidden layer.
* Added more hidden layers.
* Used different activation functions for the hidden layers.
* Added or reduced the number of epochs to the training regimen.


1. Created a new Jupyter Notebook file and named it `AlphabetSoupCharity_Optimization.ipynb`.

2. Imported the dependencies and read in the `charity_data.csv` to a Pandas DataFrame.

3. Preprocessed the dataset as you did in Step 1. Was sure to adjust for any modifications that came out of optimizing the model.

4. Designed a neural network model, and was sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

5. Saved and exported the results to an HDF5 file. Name the file `charity_model_Optimization.h5`.

### Step 4: Wrote a Report on the Neural Network Model

For this part of the assignment, I wrote a report on the performance of the deep learning model you created for Alphabet Soup.

The report contained the following:

1. **Overview** of the analysis: Explained the purpose of this analysis.

2. **Results**: Used bulleted lists and images to support my answers, addressed the following questions:

  * Data Preprocessing
    * What variable(s) are the target(s) for your model?
    * What variable(s) are the features for your model?
    * What variable(s) should be removed from the input data because they are neither targets nor features?

* Compiled, Trained, and Evaluated the Model
    * How many neurons, layers, and activation functions did you select for your neural network model, and why?
    * Were you able to achieve the target model performance?
    * What steps did you take in your attempts to increase model performance?

3. **Summary**: Summarized the overall results of the deep learning model. Included a recommendation for how a different model could solve this classification problem, and then explained my recommendation.

### References

IRS. Tax Exempt Organization Search Bulk Data Downloads. [https://www.irs.gov/](https://www.irs.gov/charities-non-profits/tax-exempt-organization-search-bulk-data-downloads)

