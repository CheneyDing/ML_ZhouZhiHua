import numpy as np
import matplotlib as plt
import pandas as pd


class Node:
    '''
    definition of desicion tree node class

    attr: attribution as parent for a new branching
    label: class label(the majority of the current sample class)
    attr_down: dict:{key, value}
            key:    categoric: attr_value
                    continuous: '<=div_value' for small part
                                '>div_value' for big part
            value:  children(node class)
    '''
    def __init__(self, attr_init=None, label_init=None, attr_down_init={}):
        self.attr = attr_init
        self.label = label_init
        self.attr_down = attr_down_init


def loadDataSet(file_name):
    '''
    :param file_name: the path of the file
    :return: the pandas dataframe of the dataset
    '''
    data_file_encode = "gb18030"
    df = pd.read_csv(file_name, encoding=data_file_encode)
    return df


def GetLabelCount(df):
    '''
    :param df: the pandas dataframe of the dataset
    :return: dist:{key, value}
             key: label
             value: label count
    '''
    label_arr = df[df.columns[-1]]
    label_count = {}
    for label in label_arr:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
    return label_count


def SameAttrValue(df):
    '''
    :param df: the pandas dataframe of the dataset
    :return: True if all row has the same attribute value, otherwise False
    '''
    for i in len(df):
        if not (df.iloc[0, 1:-1] == df.iloc[i, 1:-1]).all():
            return False
    return True


def InfoEnt(label_arr):
    '''
    :param label_arr: the label array of the dataset
    :return: the information entropy of the dataset
    '''
    ent = 0
    n = len(label_arr)
    label_count = GetLabelCount(label_arr)
    for label in label_count:
        ent -= (label_count[label] / n) * np.log2(label_count[label] / n)
    return ent


def InfoGain(df, attr):
    '''
    :param df: the pandas dataframe of the dataset
    :param attr: the specify attribute
    :return: information gain of df divided by attribute
    '''
    info_gain = InfoEnt(df[df.columns[-1]])
    n = len(df[attr])
    if df[attr].dtype == (float, int):
        sub_info_ent = {}
        df = df.sort([attr], asending=1)
        value_arr = df[attr]
        label_arr = df[df.columns[-1]]
        for i in range(n):
            div = (value_arr[i] + value_arr[i+1]) / 2
            sub_info_ent[div] = ((i+1) / n * InfoEnt(label_arr[0:i+1]) \
                                 + (n-i-1) / n * InfoEnt(label_arr[i+1:-1]))
        div_value, min_sub_info_ent = min(sub_info_ent.items(), key=lambda x:x[1])
        info_gain -= min_sub_info_ent
    else:
        value_count = GetValueCount(df[attr])
        for value in value_count:
            df_v = df[df[attr].isin[value]]
            info_gain -= value_count[value] / len(df[attr]) * InfoEnt(df_v[df.columns[-1]])

    return info_gain, div_value


def SelectOptAttr(df):
    '''
    :param df: the pandas dataframe of the dataset
    :return: (the optimal attribute, the optimal divide value)
    '''
    gain = 0
    # for each attribute, compute info gain, and find the attribute which has maximum info gain
    for attr in df.columns[1:-1]:
        # compute info gain
        gain_tmp, div_tmp = InfoGain(df, attr)
        if gain_tmp > gain:
            gain = gain_tmp
            opt_attr = attr
            div_val = div_tmp
    return opt_attr, div_val


def GetValueCount(value_arr):
    '''
    :param value_arr: the attribute's value array
    :return: dist:{key, value}
             key: value name
             value: value count
    '''
    value_count = {}
    for value in value_arr:
        if value in value_count:
            value_count[value] += 1
        else:
            value_count[value] = 1
    return value_count

def TreeGenerate(df):
    # generate node
    node = Node(None, None, {})
    # if samples in set D are the same class C, set leaf node the Class C and return
    label_count = GetLabelCount(df[df.columns[-1]])
    if len(label_count) == 1:
        node.label = max(label_count, key=label_count.get())
        return node
    # if attribute set A is NULL or the samples in D are the same value on attribute A,
    # set leaf node the class which has the most samples on set D
    if len(df.columns[1:-1]) == 0 or SameAttrValue(df):
        node.label = max(label_count, key=label_count.get())
        return node
    # select optimal divide attribute
    opt_attr, div_val = SelectOptAttr(df)
    node.attr = opt_attr
    if div_val == 0: # categoric variable
        # for each attribute value in a, generate a branch
        value_count = GetValueCount(df[opt_attr])
        for v in value_count:
            # get D_v by v
            df_v = df[df[opt_attr].isin[v]]
            # if the sample numbers of D_v is zero, set node the class which has the most samples in D
            if df_v.empty:
                node.attr_down[v] = Node(None, max(label_count, key=label_count.get()), {})
            # remove optimal attribute and generate tree based on D_v
            else:
                df_v = df_v.drop(opt_attr, 1)
                node.attr_down[v] = TreeGenerate(df_v)
    else: # continuous variable
        value_l = "<=%.3f" % div_val
        value_r = ">%.3f" % div_val
        df_v_l = df[df[opt_attr] <= div_val]
        df_v_r = df[df[opt_attr] > div_val]
        node.attr_down[value_l] = TreeGenerate(df_v_l)
        node.attr_down[value_r] = TreeGenerate(df_v_r)
    # return tree
    return node

if __name__ == "__main__":
    loadDataSet("../data/watermelon_3.csv")