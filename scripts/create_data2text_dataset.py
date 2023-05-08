from datasets import load_dataset, Dataset, DatasetDict
import json
import pandas as pd
from tqdm import tqdm
import numpy as np

"""
Code adapted from PLOG
"""


month_map = {'january': 1, 'february':2, 'march':3, 'april':4, 'may':5, 'june':6,
                 'july':7, 'august':8, 'september':9, 'october':10, 'november':11, 'december':12, 
                 'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}

### regex list

# number format: 
'''
10
1.12
1,000,000
10:00
1st, 2nd, 3rd, 4th
'''
pat_num = r"([-+]?\s?\d*(?:\s?[:,.]\s?\d+)+\b|[-+]?\s?\d+\b|\d+\s?(?=st|nd|rd|th))"

pat_add = r"((?<==\s)\d+)"

# dates
pat_year = r"\b(\d\d\d\d)\b"
pat_day = r"\b(\d\d?)\b"
pat_month = r"\b((?:jan(?:uary)?|feb(?:ruary)?|mar(?:rch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?))\b"

### return processed table to compute ranks of cells
def process_num_table(t, col):
    # dates
    date_pats = t[col].str.extract(pat_month, expand=False)
    if not date_pats.isnull().all():
        year_list = t[col].str.extract(pat_year, expand=False)
        day_list = t[col].str.extract(pat_day, expand=False)
        month_list = t[col].str.extract(pat_month, expand=False)
        month_num_list = month_list.map(month_map)

        # pandas at most 2262
        year_list = year_list.fillna("2260").astype("int")
        day_list = day_list.fillna("1").astype("int")
        month_num_list = month_num_list.fillna("1").astype("int")
        try:
            date_series = pd.to_datetime(pd.DataFrame({'year': year_list, 'month': month_num_list, 'day': day_list}))
            return date_series
        except:
            pass

    # mixed string and numerical
    pats = t[col].str.extract(pat_add, expand=False)
    if pats.isnull().all():
        pats = t[col].str.extract(pat_num, expand=False)
    if pats.isnull().values.any():
        raise ExeError()
    nums = pats.str.replace(",", "")
    nums = nums.str.replace(":", "")
    nums = nums.str.replace(" ", "")
    try:
        nums = nums.astype("float")
    except:
        nums = nums.str.replace(".", "")
        nums = nums.astype("float")

    return nums

def add_agg_cell(t, col):
  '''
  sum or avg for aggregation
  '''

  # unused
  if t.dtypes[col] == np.int64 or t.dtypes[col] == np.float64:
    sum_res = t[col].sum()
    avg_res = t[col].mean()
    return sum_res, avg_res
  else:
    pats = t[col].str.extract(pat_add, expand=False)
    if pats.isnull().all():
      pats = t[col].str.extract(pat_num, expand=False)
    if pats.isnull().all():
      raise ExeError
    pats.fillna("0.0")
    nums = pats.str.replace(",", "")
    nums = nums.str.replace(":", "")
    nums = nums.str.replace(" ", "")
    try:
      nums = nums.astype("float")
    except:
      nums = nums.str.replace(".", "")
      nums = nums.astype("float")

    # print (nums)

    return nums.sum(), nums.mean()

def precompute_cell_information(pd_table):
    cell_ranks = {}
    columns = pd_table.columns
    for i in range(pd_table.shape[1]):
        try:
            sub_table = process_num_table(pd_table, columns[i])
            ranks = sub_table.rank(method='min')
        except:
            for j in range(pd_table.shape[0]):
                cell_ranks[(j, i)] = None, None
            continue

        rank_flag = not sub_table.isnull().values.any()
        for j in range(pd_table.shape[0]):
            if rank_flag:
                max_rank = len(ranks) - int(ranks.iloc[j]) + 1
                min_rank = -int(ranks.iloc[j])
            else:
                max_rank, min_rank = None, None
            cell_ranks[(j, i)] = max_rank, min_rank
    agg_cells = {}
    for i in range(pd_table.shape[1]):
        try:
            sum_val, avg_val = add_agg_cell(pd_table, columns[i])
        except:
            continue
        agg_cells[i] = (sum_val, avg_val)

    return cell_ranks, agg_cells

def process_logicnlg(data_file, ids, serialize_order='column', pre_com=True):
    '''
    Args:
        data_file: path to data file
        ids: table ids
        serialize_order: row-wise or column-wise serialization
        pre_com: whether to do numerical pre-computation
    Returns:
       new data with serialized input
    '''
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        new_data = []
        for table_id in tqdm(ids):
            entry = data[table_id]
            table = pd.read_csv('logicnlg/all_csv/' + table_id, sep='#')
            if pre_com:
                cell_ranks, agg_cells = precompute_cell_information(table)
            columns = table.columns
            for e in entry:
                doc = {}
                # "sent" can be either logic_str or text, depending on the dataset
                doc['sent'] = e[0]
                doc['table_id'] = table_id
                src_text = "<table> " + "<caption> " + e[2] + " </caption> "
                tmp = ""
                if serialize_order == 'column':
                    # column-wise serialization
                    for col in e[1]:
                        for i in range(len(table)):
                            if isinstance(table.iloc[i][columns[col]], str):
                                entity = map(lambda x: x.capitalize(), table.iloc[i][columns[col]].split(' '))
                                entity = ' '.join(entity)
                            else:
                                entity = str(table.iloc[i][columns[col]])
                            if pre_com:
                                max_rank, min_rank = cell_ranks[(i, col)]
                                if max_rank is not None:
                                    tmp += f"<cell> {entity} <col_header> {columns[col]} </col_header> <row_idx> {i} </row_idx> <max_rank> {max_rank} </max_rank> <min_rank> {min_rank} </min_rank> </cell> "
                                else:
                                    tmp += f"<cell> {entity} <col_header> {columns[col]} </col_header> <row_idx> {i} </row_idx> </cell> "
                            else:
                                tmp += f"<cell> {entity} <col_header> {columns[col]} </col_header> <row_idx> {i} </row_idx> </cell> "

                        if pre_com and agg_cells and agg_cells.get(col, None):
                            sum_val, avg_val = agg_cells[col]
                            src_text += f"<sum_cell> {sum_val} <col_header> {columns[col]} </col_header> </sum_cell> "
                            src_text += f"<avg_cell> {avg_val} <col_header> {columns[col]} </col_header> </avg_cell> "

                elif serialize_order == 'row':
                    # Row-wise seralization
                    for i in range(len(table)):
                        for col in e[1]:
                            if isinstance(table.iloc[i][columns[col]], str):
                                entity = map(lambda x: x.capitalize(), table.iloc[i][columns[col]].split(' '))
                                entity = ' '.join(entity)
                            else:
                                entity = str(table.iloc[i][columns[col]])
                            if pre_com:
                                max_rank, min_rank = cell_ranks[(i, col)]
                                if max_rank is not None:
                                    tmp += f"<cell> {entity} <col_header> {columns[col]} </col_header> <row_idx> {i} </row_idx> <max_rank> {max_rank} </max_rank> <min_rank> {min_rank} </min_rank> </cell> "
                                else:
                                    tmp += f"<cell> {entity} <col_header> {columns[col]} </col_header> <row_idx> {i} </row_idx> </cell> "
                            else:
                                tmp += f"<cell> {entity} <col_header> {columns[col]} </col_header> <row_idx> {i} </row_idx> </cell> "

                    for col in e[1]:
                        if pre_com and agg_cells and agg_cells.get(col, None):
                            sum_val, avg_val = agg_cells[col]
                            src_text += f"<sum_cell> {sum_val} <col_header> {columns[col]} </col_header> </sum_cell> "
                            src_text += f"<avg_cell> {avg_val} <col_header> {columns[col]} </col_header> </avg_cell> "
                src_text += tmp + "</table>"
                doc['src_text'] = src_text
                new_data.append(doc)
        return new_data


dataset_dict = DatasetDict()
for split in ["train","val","test"]:
    ids = json.load(open("logicnlg/{}_ids.json".format(split)))
    data = process_logicnlg("logicnlg/{}_lm.json".format(split), ids,"row")

    dataset = {"table_id": [], "src_text": [], "sent": []}
    for dat in data:
        for k,v in dat.items():
            dataset[k].append(v)
    dataset_dict[split] = Dataset.from_dict(dataset)

print(dataset_dict)

dataset_dict.save_to_disk("data/LogicNLG")