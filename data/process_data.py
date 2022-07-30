import argparse

import pandas as pd
from sqlalchemy import create_engine

parser = argparse.ArgumentParser()
parser.add_argument(
    "-messages-filepath",
    required=True,
    help="Absolute or relative path to the messages file."
)
parser.add_argument(
    "-categories-filepath",
    required=True,
    help="Absolute or relative path to the categories file."
)
parser.add_argument(
    "-database-filepath",
    required=True,
    help="Absolute or relative path to the Sqlite database the data will be loaded to."
)


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    Function that loads the data from the files as a DataFrames, and it merges them.

    @param messages_filepath: path to the messages' dataset
    @param categories_filepath: path to the categories' dataset
    @return:
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, on="id", how="left")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function that takes in the merged DataFrame and cleans it.

    @param df: The merged DataFrame.
    @return: A cleaned DataFrame.
    """

    categories = df["categories"].str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.transform(lambda item: item[:-2]).values.tolist()
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    df = df.drop(columns=["categories"])
    df = pd.concat([df, categories], axis=1)

    # First, drop duplicates at the message level.
    df = df.drop_duplicates(subset=["message"])
    # After, as a safety net, drop duplicates at the whole dataset level.
    df = df.drop_duplicates()

    df = filter_constant_columns(df, verbose=True)

    # Drop row with related == 2, as they represent exceptions. We want all the targets to be binary 0/1.
    df = df[df["related"] != 2]

    return df


def filter_constant_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Function that filters out all the constant columns from the DataFrame.
    A constant column is neither good as a feature nor as a target.

    @param df: Data DataFrame.
    @param verbose: Whether to print the filtered columns.
    @return: A DataFrame with the filtered columns.
    """

    # Find all the constant columns within the dataset.
    constant_columns = []
    for column in df.columns:
        column_values = df[column]
        if len(pd.unique(column_values)) <= 1:
            constant_columns.append(column)

    # Drop the constant columns from the dataset.
    df = df.drop(columns=constant_columns)
    if verbose is True:
        print("\tThe following columns are constant. Therefore, are dropped: {}".format(constant_columns))

    return df


def save_data(df: pd.DataFrame, database_filename: str):
    """
    Function that saves the cleaned DataFrame to a SQLite database.

    @param df: The processed DataFrame.
    @param database_filename: SQLITE database filepath.
    @return: None
    """

    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('processed', engine, index=False)


def main():
    """
    Main function that aggregates all the logic.

    @return: None
    """

    args = parser.parse_args()

    messages_filepath = args.messages_filepath
    categories_filepath = args.categories_filepath
    database_filepath = args.database_filepath

    print('Loading data...\n\tMESSAGES: {}\n\tCATEGORIES: {}'
          .format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...\n\tDATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)

    print('Cleaned data saved to database!')


if __name__ == '__main__':
    main()
