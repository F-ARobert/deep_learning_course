import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

########################################################################
# Prepare data
########################################################################
dataset = pd.read_csv("life_expectancy.csv")

dataset = dataset.drop(['Country'], axis=1)  # removes Country column from consideration

# print(dataset.head())
# print(dataset.describe())

labels = dataset.iloc[:, -1]  # selects all rows and last column
features = dataset.iloc[:, 0:-1]  # selects all rows and all columns save for the last one

features = pd.get_dummies(features)  # convert categorical data into numerical

# print(features.head())
# print(features.describe())

# Split training data and test data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.15,
                                                                            random_state=30)

# Standardize/normalize data
# I'm using standardization

# List all numerical features. This allos to select numerical features (float64 or int64) automatically
numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns

# Creating the transformer that will be applied to the data
ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')

# Apply fit column transformer to training data
features_train_scaled = ct.fit_transform(features_train)

# Transform test data using ct columntransformer instance
features_test_scaled = ct.transform(features_test)

########################################################################
# Create model
########################################################################
# Creating a sequential model instance
my_model = Sequential()

# Create input layer to the network model
input = InputLayer(input_shape=(features.shape[1],))

# Add input layer to model
my_model.add(input)

# Add a hidden layer with 128 units with relu activation
my_model.add(Dense(32, activation="relu"))

# Add an outpu layer with a single neure (you need a single output for a regression prediction)
my_model.add(Dense(1))

print(my_model.summary())

########################################################################
# Initialize optimizer and compile model
########################################################################
# Create instance of Adam
opt = Adam(learning_rate=0.01)

# Compile model
my_model.compile(loss='mse', metrics=['mae'], optimizer=opt)

########################################################################
# Fit and evaluate the model
########################################################################
# Train model
my_model.fit(features_train_scaled, labels_train, epochs=40, batch_size=1, verbose=1)

# Evaluate model
res_mse, res_mae = my_model.evaluate(features_test_scaled, labels_test, verbose=0)

# Print results
print("Final loss (RMSE) = ", res_mse)
print("Final metric (MAE) = ", res_mae)
