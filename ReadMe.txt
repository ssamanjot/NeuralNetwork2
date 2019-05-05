The task is to predict the final student scores in a high school course. The attributes are student grades on 2 multiple choice assignments M1 and M2, 2 programming assignments P1 and P2, and the final exam F. The scores of all the components are integers in the range [0,100]. All the attributes are multivalued discrete. Again, check the csv files to see the attribute values. The final output is also integer with a range [0,100]
For performing above task, we are using both Gradient descent and Stochastic gradient descent

We have been provided with attributes and labels split into training and development data in files music*.csv
• education_train.csv: attributes for training data, without labels, with
column names on the first line
• education_train_keys.txt: labels for training data, as a single
column with no column names
• education_dev.csv: attributes for development data, without labels, with
column names on the first line; shares the same format (but not necessarily the same number
of records) as the test set that you will be evaluated on
• education_dev_keys.txt: labels for development data, as a single
column with no column names