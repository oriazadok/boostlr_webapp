import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

import jpype
from jpype import JClass

from BoostingLRWrapper import BoostingLRWrapper
from utils import kendalls_tau

def start_jvm():
    # Start the JVM and set up the classpath
    jvm_args = ["-Xmx1g"]  # Set maximum heap size for JVM
    cp = ["./", "./lib/*", "./weka"]  # Set the classpath for Weka and other Java dependencies
    jpype.startJVM(*jvm_args, classpath=cp, convertStrings=True)

def stop_jvm():
    jpype.shutdownJVM()

def load_xarff(file_path):
    """Custom loader to read XARFF files and convert them into a pandas DataFrame."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Separate metadata and data
    data_start_idx = lines.index('@data\n') + 1
    metadata = lines[:data_start_idx]
    data_lines = lines[data_start_idx:]

    # Extract column names and types from metadata
    column_names = []
    attribute_info = {}
    for line in metadata:
        if line.startswith('@attribute'):
            # Extract the column name and its type information
            match = re.findall(r'@attribute\s+([\w\d_-]+)\s+(.*)', line)
            if match:
                column_name, attribute_type = match[0]
                column_names.append(column_name)
                attribute_info[column_name] = attribute_type.strip()

    # Parse the data lines into a list of lists
    data = []
    for line in data_lines:
        # Split the line by commas and strip any extra spaces
        split_line = re.split(r',\s*(?![^{}]*\})', line.strip())
        # Handle the RANKING column by converting it to a list
        if split_line[-1].startswith('{'):
            ranking = split_line[-1][1:-1].split('>')
            split_line[-1] = ranking
        data.append(split_line)

    # Create a DataFrame from the parsed data
    df = pd.DataFrame(data, columns=column_names)

    return df, attribute_info

def save_to_xarff(df, file_path, relation_name="dataset", attribute_info=None):
    """Save a Pandas DataFrame as a XARFF file."""
    with open(file_path, 'w') as f:
        # Write the relation name
        f.write(f"@relation {relation_name}\n\n")

        # Write the attribute names and types using the provided attribute_info
        for column in df.columns:
            attribute_type = attribute_info.get(column, 'STRING')
            f.write(f"@attribute {column} {attribute_type}\n")

        # Write the data
        f.write("\n@data\n")
        for _, row in df.iterrows():
            row_data = []
            for value in row:
                if isinstance(value, list):
                    # Convert list back to ranking format
                    value = '>'.join(value)
                row_data.append(value)
            f.write(','.join(map(str, row_data)) + '\n')

def load_dataset_as_Instances(file):
    DataSource = JClass("weka.core.converters.ConverterUtils$DataSource")
    data = DataSource.read(file)
    data.setClassIndex(data.numAttributes() - 1)
    return data

def run_lrt(train_data, test_data):
    total_kt = 0.0
    lrt = JClass("weka.classifiers.labelranking.LRT")()
    lrt.buildClassifier(train_data)
    for i in range(test_data.numInstances()):
        instance = test_data.instance(i)
        prefs = model.boosting_lr.preferences(instance)
        preds = lrt.distributionForInstance(instance)

        total_kt += kendalls_tau(preds, prefs)

    # Calculate and print the average Kendall's Tau
    accuracy = total_kt / test_data.numInstances()
    print(f"Test Accuracy (Average Kendall's Tau): {accuracy * 100:.2f}%")

if __name__ == "__main__":

    start_jvm()

    dataset = "./datasets/sushi_mini_same_rank.xarff"

    # Extract the base name (without extension) and directory
    base_name = os.path.basename(dataset).replace(".xarff", "")
    directory = os.path.dirname(dataset)

    # Create the full paths and the desired string
    train_base_name = f"{base_name}_train"
    test_base_name = f"{base_name}_test"

    train_dataset = os.path.join(directory, f"{train_base_name}.xarff")
    test_dataset = os.path.join(directory, f"{test_base_name}.xarff")

    # Load the dataset and get attribute information
    df, attribute_info = load_xarff(dataset)

    total_accuracy = 0
    splits = [0.1, 0.2, 0.3, 0.4, 0.5]
    for split in splits:

        # Split the data into training and test sets using Pandas
        train_data, test_data = train_test_split(df, test_size=split, random_state=42)

        # Save the split datasets to XARFF files with the original attribute info
        save_to_xarff(train_data, train_dataset, relation_name=train_base_name, attribute_info=attribute_info)
        save_to_xarff(test_data, test_dataset, relation_name=test_base_name, attribute_info=attribute_info)

        print("Training and test datasets saved to XARFF files.")

        print("train_dataset: ", train_dataset)
        # Load dataset as Instances
        train_data = load_dataset_as_Instances(train_dataset)
        test_data = load_dataset_as_Instances(test_dataset)

        # run_lrt(train_data, test_data)

        # Initialize the BoostingLR wrapper
        model = BoostingLRWrapper(max_iterations=50, seed=7)

        # Train the model on the training dataset
        model.fit(train_data)

        # Predict on the test dataset
        predictions = model.predict(test_data)

        # Evaluate predictions
        total_kt = 0.0
        for i in range(test_data.numInstances()):
            instance = test_data.instance(i)
            prefs = model.boosting_lr.preferences(instance)
            preds = predictions[i]

            total_kt += kendalls_tau(preds, prefs)

        # Calculate and print the average Kendall's Tau
        accuracy = total_kt / test_data.numInstances()
        print(f"Test Accuracy (Average Kendall's Tau): {accuracy * 100:.2f}%")
        total_accuracy += accuracy

    print(f"total avg:  {(total_accuracy / len(splits)) * 100:.2f}%")

    # Shutdown the JVM
    stop_jvm()

