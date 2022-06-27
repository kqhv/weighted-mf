import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.preprocessing import LabelEncoder


def read_data(filepath: str):
  """
  Specific for movielens-100K
  """

  df = pd.read_csv(filepath, sep='\t', names=['user', 'item', 'rating', 'timestamp'])
  df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.floor('d')
  df = df.sort_values('date', ascending=False)

  return df


def df2mat(df):
  """
  [user, item, rating] to coo matrix.
  """

  assert {'user', 'item', 'rating'}.issubset(df.columns)

  ule, ile = LabelEncoder(), LabelEncoder()
  df['user'] = ule.fit_transform(df['user'])
  df['item'] = ile.fit_transform(df['item'])

  row = df['user']
  col = df['item']
  data = df['rating']
  mat = coo_matrix((data, (row, col)))

  return mat, ule, ile
