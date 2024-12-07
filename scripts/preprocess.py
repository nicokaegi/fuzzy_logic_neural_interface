import pandas as pd
import numpy as np
import os
from glob import glob

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#####
# Data Placeholder
data_dir = "../data/eeg_muse2_moto_imagery_brain_ electrical_activity/"

output_file_path = "../data/preprocessed_motor_imagery_data.csv"

######
# Data needed for processing

# Colunms of the dataset to work with (File_ID - is build by us)
relevant_columns = [
    "File_ID",
    "TimeStamp",
    "Delta_TP9",
    "Delta_AF7",
    "Delta_AF8",
    "Delta_TP10",
    "Theta_TP9",
    "Theta_AF7",
    "Theta_AF8",
    "Theta_TP10",
    "Alpha_TP9",
    "Alpha_AF7",
    "Alpha_AF8",
    "Alpha_TP10",
    "Beta_TP9",
    "Beta_AF7",
    "Beta_AF8",
    "Beta_TP10",
    "Gamma_TP9",
    "Gamma_AF7",
    "Gamma_AF8",
    "Gamma_TP10",
    "Elements",
]

# In build motor imagery markers mapping
motor_imagery_mapping = {
    "/Marker/1": "Left",
    "/Marker/2": "Right",
    "/Marker/3": "Relax",
    "/muse/elements/blink": "Blink",
    "/muse/elements/jaw_clench": "Jaw_Clench",
    "/Marker/4": "Marker_4",
}

# EEG Bands we work with
eeg_bands = [
    "Delta_TP9",
    "Delta_AF7",
    "Delta_AF8",
    "Delta_TP10",
    "Theta_TP9",
    "Theta_AF7",
    "Theta_AF8",
    "Theta_TP10",
    "Alpha_TP9",
    "Alpha_AF7",
    "Alpha_AF8",
    "Alpha_TP10",
    "Beta_TP9",
    "Beta_AF7",
    "Beta_AF8",
    "Beta_TP10",
    "Gamma_TP9",
    "Gamma_AF7",
    "Gamma_AF8",
    "Gamma_TP10",
]

# Task labels we will use for classification
task_labels = ["Left", "Right", "Relax"]

###########
# Functions


def load_data(file_path, file_id):
    """
    Load data from a CSV file and add a File_ID column.

    Args:
        file_path (str): The path to the CSV file.
        file_id (str): The identifier to be added as a new column in the
                       DataFrame.

    Returns:
        pandas.DataFrame: The loaded DataFrame with an additional File_ID
                          column.
    """
    df = pd.read_csv(file_path)
    # Set the id of the file
    df["File_ID"] = file_id

    return df


def df_relevant_columns(df, columns):
    """
    Selects and returns specific columns from a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to select columns from.
    columns (list of str): A list of column names to be selected from the
                           DataFrame.

    Returns:
    pandas.DataFrame: A DataFrame containing only the specified columns.
    """
    return df[columns]


def create_task_labels(df, label_mapping):
    """
    Maps task labels to elements in the given DataFrame and removes the
    original elements column.
    Args:
        df (pandas.DataFrame): The input DataFrame containing an
                               'Elements' column.
        label_mapping (dict): A dictionary mapping elements to task labels.
    Returns:
        pandas.DataFrame: The modified DataFrame with a new 'Task_Labels'
                          column and without the 'Elements' column.
    """
    # Map tasks label to elements
    df["Task_Labels"] = df["Elements"].map(label_mapping)
    # Remove elements col
    df = df.drop(columns=["Elements"])

    return df


def add_event_count(df):
    """
    Adds an "Event_Count" column to the DataFrame, which contains a count
    of occurrences of each unique task label in the "Task_Labels" column,
    formatted as a string with a 3-digit number.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing a "Task_Labels" column.

    Returns:
    pd.DataFrame: The modified DataFrame with an additional
                  "Event_Count" column.

    Example:
    >>> data = pd.DataFrame({"Task_Labels": ["Left", "Left", "Right",
                                            "Right", "Right", np.nan, "Left"]})
    >>> add_event_count(data)
       Task_Labels Event_Count
    0        Left      Left_001
    1        Left      Left_002
    2       Right     Right_001
    3       Right     Right_002
    4       Right     Right_003
    5         NaN           NaN
    6        Left      Left_003
    """
    event_counter = {}  # Dictionary to keep track of the count task label
    event_count_column = []  # List to store the new event count values

    for label in df["Task_Labels"]:
        if pd.notna(label):
            event_counter[label] = event_counter.get(label, 0) + 1
            event_count_column.append(f"{label}_{event_counter[label]:03d}")
        else:
            event_count_column.append(np.nan)

    df["Event_Count"] = event_count_column
    return df


def propagate_task_labels_and_event_counts(df):
    """
    Propagates task labels and event counts in a DataFrame.

    This function performs the following steps:
    1. Removes rows before the first valid task label.
    2. Propagates (forward fills) the 'Task_Labels' and 'Event_Count' columns.

    Parameters:
    df (pandas.DataFrame): The input DataFrame with columns
    'Task_Labels' and 'Event_Count'.

    Returns:
    pandas.DataFrame: The modified DataFrame with propagated task labels and
                      event counts.
    """
    # TODO: Add logger to explain row removieng
    # Remove row before the first task label
    first_label_index = df["Task_Labels"].first_valid_index()
    df = df.loc[first_label_index:].reset_index(drop=True)

    # Propagate Task_Labels and Event_Count
    df["Task_Labels"] = df["Task_Labels"].ffill()
    df["Event_Count"] = df["Event_Count"].ffill()

    return df


def drop_event_label_row(df, eeg_band_columns):
    """
    Drops rows from the DataFrame where all specified EEG band columns contain
    NaN values.

    This function removes rows where all the columns specified
    in `eeg_band_columns` are NaN, and then resets the index of the DataFrame.
    It also prints the number of rows with NaN values before and after the
    operation, as well as the number of rows removed.

    Args:
        df (pandas.DataFrame): The input DataFrame containing EEG data.
        eeg_band_columns (list of str): List of column names representing
                                        EEG bands to check for NaN values.

    Returns:
        pandas.DataFrame: The DataFrame with rows dropped where all specified
                          EEG band columns were NaN.
    Logs:
        NaN rows before event rows cleaning: The number of rows with NaN values
                                             before cleaning.
        Remaining NaN rows to interpolate: The number of rows with NaN values
                                           after cleaning.
        Removed NaN rows: The number of rows removed during cleaning.
    """
    rows_with_na_before = df[eeg_band_columns].isnull().any(axis=1).sum()
    df = df.dropna(subset=eeg_band_columns, how="all").reset_index(drop=True)
    rows_with_na_after = df[eeg_band_columns].isnull().any(axis=1).sum()

    logger.info(f"NaN rows before event rows cleaning: {rows_with_na_before}")
    logger.info(f"Remaining NaN rows to interpolate: {rows_with_na_after}")
    logger.info(
        f"""Removed NaN rows:
                {rows_with_na_before - rows_with_na_after}
        """
    )
    logger.info("")

    return df


def interpolate_nan_points(df, eeg_band_columns):
    """
    Interpolates NaN values in specified columns of a DataFrame using
    linear interpolation.
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
                          eeg_band_columns (list of str): List of column
                          names in which to interpolate NaN values.
    Returns:
    pandas.DataFrame: The DataFrame with NaN values
                      interpolated in the specified columns.
    Logs:
    NaN rows before interpolation: The number of rows with NaN values
                                   before interpolation.
    Interpolated NaN rows: The number of NaN rows that were interpolated.
    Remaining NaN rows - must be 0: The number of rows with NaN values
                                    after interpolation (should be 0).
    """
    rows_with_na_before = df[eeg_band_columns].isnull().any(axis=1).sum()
    df[eeg_band_columns] = df[eeg_band_columns].interpolate(method="linear", axis=0)
    rows_with_na_after = df[eeg_band_columns].isnull().any(axis=1).sum()

    logger.info(f"NaN rows before interpolation: {rows_with_na_before}")
    logger.info(
        f"""
            Interpolated NaN rows: {rows_with_na_before - rows_with_na_after}
        """
    )
    logger.info(f"Remaining NaN rows - must be 0: {rows_with_na_after}")
    logger.info("")

    return df


def timestamp_to_datetime(df):
    """
    Converts the 'TimeStamp' column in a DataFrame to datetime format.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing a 'TimeStamp' column.

    Returns:
    pandas.DataFrame: The DataFrame with the 'TimeStamp' column converted to datetime format.
    """
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])

    return df


def handle_time_synchronization(df, eeg_band_columns):
    """
    Handle time synchronization by removing duplicates and aggregating EEG
    band data.

    This function processes a DataFrame to handle time synchronization
    issues by:
    1. Identifying non-duplicate rows based on "TimeStamp" and "Event_Count".
    2. Aggregating EEG band data by computing the mean for
       each group of duplicates.
    3. Combining the aggregated data with non-duplicates, removing duplicates,
       and sorting by "TimeStamp".

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data to be processed.
    eeg_bands (list of str): A list of column names representing the EEG bands
                             to be aggregated.

    Returns:
    pd.DataFrame: A DataFrame with duplicates removed,
                  EEG band data aggregated, and sorted by "TimeStamp".
    """
    # Identify the non duplicates for each TimeStamp and Event_Count
    non_duplicates = df[~df.duplicated(subset=["TimeStamp", "Event_Count"], keep=False)]
    # Compute the mean for the eeg_band_columns for each group of duplicates
    aggregated_data = df.groupby(
        ["TimeStamp", "Event_Count", "Task_Labels", "File_ID"], as_index=False
    )[eeg_band_columns].mean()
    # Combination of aggregated data and non-duplicates,
    # with duplicates removed and sorted by TimeStamp.
    df = (
        pd.concat([aggregated_data, non_duplicates])
        .drop_duplicates(subset=["TimeStamp", "Event_Count"])
        .sort_values(by="TimeStamp")
    )

    return df


def filter_rows_by_labels(df, labels):
    """
    Filters the rows of a DataFrame based on specified task labels.

    Parameters:
    df (pandas.DataFrame): The DataFrame to filter.
    task_labels (list of str, optional): A list of task labels to filter by.

    Returns:
    pandas.DataFrame: The filtered DataFrame containing only rows with the
                      specified task labels.
    """
    df = df[df["Task_Labels"].isin(labels)]
    # Display the count of each label to confirm filtering and show a sample
    print(df["Task_Labels"].value_counts())

    return df


def preprocess_file(file_path, file_id, motor_imagery_marker_mapping):
    """
    Preprocess the given file by performing a series of data manipulation
    steps.
    Parameters:
    file_path (str): The path to the file to be processed.
    file_id (str): The unique identifier for the file.
    motor_imagery_marker_mapping (dict): A dictionary mapping motor imagery
                                         markers to task labels.
    Returns:
    pandas.DataFrame: The preprocessed data.
    Steps:
    1. Load the data from the specified file.
    2. Select relevant columns from the data.
    3. Replace elements to create task labels based on motor imagery markers.
    4. Add an event count to the data.
    5. Propagate task labels and event counts through the data.
    6. Remove empty rows from the data.
    7. Interpolate missing points in the data based on time.
    8. Handle time synchronization issues in the data.
    9. Filter rows based on specific task labels ('Left', 'Right', 'Relax').
    Logging:
    Logs the progress of each preprocessing step.
    """
    logger.info(
        f"""
                Starting preprocessing for file: {file_path} with ID: {file_id}"""
    )

    # Step 1 - Load the data
    df = load_data(file_path, file_id)
    logger.info("Data loaded successfully.")

    # Step 2 - Select relevant columns
    df = df_relevant_columns(df, relevant_columns)
    logger.info("Relevant columns selected.")

    # Step 3 - Replace elements for task labels
    df = create_task_labels(df, motor_imagery_marker_mapping)
    logger.info("Task labels created.")

    # Step 4 - Add event count
    df = add_event_count(df)
    logger.info("Event count added.")

    # Step 5 - Propagate task labels and event count
    df = propagate_task_labels_and_event_counts(df)
    logger.info("Task labels and event counts propagated.")

    # Step 6 - Remove empty rows
    df = drop_event_label_row(df, eeg_bands)
    logger.info("Empty rows removed.")

    # Step 7 - Interpolate band value based on time for missing points
    df = interpolate_nan_points(df, eeg_bands)
    logger.info("NaN points interpolated.")

    # Step 8 - Handle time synchronization
    df = timestamp_to_datetime(df)
    df = handle_time_synchronization(df, eeg_bands)
    logger.info("Time synchronization handled.")

    # Step 9 - Filter rows based on Task_Labels ('Left', 'Right', 'Relax')
    df = filter_rows_by_labels(df, task_labels)
    logger.info("Rows filtered by task labels.")

    logger.info(f"Preprocessing completed successfully for the file id: {file_id}.")
    return df


def perform_sanity_check(combined_data, relevant_columns):
    """
    Perform a sanity check on the combined data.

    Parameters:
    combined_data (pandas.DataFrame): The combined DataFrame to check.
    relevant_columns (list of str): List of expected column names.

    Returns:
    None
    """
    logger.info("Performing sanity check on the combined data.")

    # Check for any remaining NaN values
    if combined_data.isnull().any().any():
        logger.warning(
            "Sanity check failed: There are still NaN values in the combined data."
        )
    else:
        logger.info("Sanity check passed: No NaN values found in the combined data.")

    # Check if all expected columns are present
    missing_columns = [
        col for col in relevant_columns if col not in combined_data.columns
    ]
    if missing_columns:
        logger.warning(
            f"Sanity check failed: Missing columns in the combined data: {missing_columns}"
        )
    else:
        logger.info(
            "Sanity check passed: All expected columns are present in the combined data."
        )

    # Check if there are any duplicate rows by single events per file with respect to timestamp
    if combined_data.duplicated(subset=["File_ID", "Event_Count", "TimeStamp"]).any():
        logger.warning(
            "Sanity check failed: There are duplicate single events in the combined data with respect to timestamp."
        )
    else:
        logger.info(
            "Sanity check passed: No duplicate single events found in the combined data with respect to timestamp."
        )


def main(
    data_directory=data_dir,
    output_file=output_file_path,
    motor_imagery_marker_mapping=motor_imagery_mapping,
):
    """
    Main function to preprocess and combine CSV files from a
    specified directory.

    Parameters:
    data_directory (str): The directory containing the CSV files to
                          be processed.
    output_file (str): The path where the combined output CSV file
                       will be saved.
    motor_imagery_marker_mapping (dict): A dictionary mapping motor
                                         imagery markers.

    Returns:
    None

    This function performs the following steps:
    1. Retrieves all CSV file paths in the specified directory.
    2. Initializes an empty list to store processed DataFrames.
    3. Iterates through each file, preprocesses it, and appends the processed DataFrame to the list.
    4. Combines all processed DataFrames into a single DataFrame.
    5. Saves the combined DataFrame to the specified output CSV file.
    6. Logs the progress and completion of the processing.
    """
    # Get all CSV file paths in the directory
    file_paths = glob(os.path.join(data_directory, "*.csv"))
    # Initialize an empty list to store processed DataFrames
    all_processed_data = []
    print(data_directory)
    # Iterate through each file and preprocess it
    for file_index, file_path in enumerate(file_paths, start=1):
        file_id = f"file_{file_index:02d}"  # Generate unique File_ID
        logger.info(f"Processing {file_path} as {file_id}...")

        # Preprocess the file
        processed_data = preprocess_file(
            file_path, file_id, motor_imagery_marker_mapping
        )

        # Append the processed DataFrame to the list
        all_processed_data.append(processed_data)

    # Combine all processed files into a single DataFrame
    combined_data = pd.concat(all_processed_data, ignore_index=True)

    # Sanity check
    perform_sanity_check(combined_data, relevant_columns)

    # Save the combined DataFrame to a CSV file
    combined_data.to_csv(output_file, index=False)

    logger.info(
        f"""
            All files processed and combined. Output saved to {output_file}.
        """
    )


main()
