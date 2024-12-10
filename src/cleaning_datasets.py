import pandas as pd 
import numpy as np




class CleaningDatasets:
    """
    Class for cleaning and transforming datasets from CSV files
    """
    def __init__(self, datapath='./data/'):
        """
        Function that initializes the class with the path to the data files
        """
        self.datapath = datapath

    # removing unnecessary columns
    def drop_columns(self, df: pd.DataFrame, columns_to_drop: list[str]) -> pd.DataFrame:
        """
        Drop specific columns from a DataFrame.
        :param df: Dataframe with columns to remove
        :param columns_to_drop: columns to be removed
        :return: dataframe without the columns to drop 
        """
        df = df.drop(columns = columns_to_drop)
        return df
    
    def drop_rows(self, df: pd.DataFrame, rows_to_drop) -> pd.DataFrame:
        """
        Drop rows based on a condition
        :param df: dataframe with rows to drop 
        :param rows_to_drop: condition to select the rows to drop 
        :return: dataframe with rows removed 
        """
        df = df.drop(df[rows_to_drop].index)
        # example : median_price = median_price.drop(median_price[(median_price['année'] != 2023)].index)
        return df
    
    def rename_columns(self, df: pd.DataFrame, names: dict[str,str]) -> pd.DataFrame:
        """
        rename columns in a dataframe
        :param df: dataframe with columns to rename
        :param names: dictionnary with old and new names
        :return: dataframe with renamed columns
        """
        df = df.rename(columns=names)
        return df

    def new_columns_sum(self, df, group, column, new_column) -> pd.DataFrame:
        sum_column = df.groupby(group)[column].sum()
        df[new_column] = df[group].map(sum_column)
        return df

    def new_columns_mean(self, df, group, column, new_column) -> pd.DataFrame:
        median_column = df.groupby(group)[column].mean()
        df[new_column] = df[group].map(median_column)
        return df
    
    def new_columns(self, df) -> pd.DataFrame:
        df['population-per-surface-district'] = df['population-district'] / df['surface-area-total']
        df['number-transactions'] = df['nb_transactions_house'] + df['nb_transactions_apartment']
        return df

    def new_columns_conditions(self, df):
        df['median-price'] = np.where(
        df['Property'] == 'House', 
        df['house-median-price'],       
        df['apartment-median-price']     
        )
        return df

    def replace_elements(self, df: pd.DataFrame, column: str, element, replaced) -> pd.DataFrame:
        df[column] = df[column].str.replace(element, replaced)
        return df

    def merging_dataset(self, df1:pd.DataFrame, df2:pd.DataFrame, column1: str, column2:str) -> pd.DataFrame:
        df = pd.merge(df1, df2, left_on=column1, right_on=column2, how='left')
        return df

    def change_type(self, df: pd.DataFrame, column: str, dtype: type) -> pd.DataFrame:
        """
        Change the type of a column
        :param df: dataframe
        :param column: target column name
        :param: desired datatype
        :return: new dataframe
        """
        df[column] = df[column].astype(dtype)
        return df









