import numpy as np
import pandas as pd
from pydotplus import graphviz

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


def GetLabelCount(label_arr):
    '''
    :param df: the pandas dataframe of the dataset
    :return: dist:{key, value}
             key: label
             value: label count
    '''
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
    for i in range(len(df)):
        if not (df.iloc[0, 1:-1] == df.iloc[i, 1:-1]).all():
            return False
    return True


def Gini(label_arr):
    label_count = GetLabelCount(label_arr)
    p_same = 0
    n = len(label_arr)
    for label in label_count:
        p_same += np.square(label_count[label] / n)
    return 1 - p_same


def GiniIndex(df, attr):
    n = len(df[attr])
    gini_index = 0
    div_value = 0
    if df[attr].dtype == float:
        sub_gini_index = {}
        for i in range(n-1):
            div = (df[attr][i] + df[attr][i+1]) / 2
            sub_gini_index[div] = ((i+1) * Gini(df[df.columns[-1]][0:i+1]) / n) \
                                    + ((n-i-1) * Gini(df[df.columns[-1]][i+1:-1]) / n)
        div_value, gini_index = min(sub_gini_index.items(), key=lambda x:x[1])
    else:
        value_count = GetValueCount(df[attr])
        for value in value_count:
           df_v = df[df[attr].isin([value])]
           gini_index += value_count[value] * Gini(df_v[df_v.columns[-1]]) / n
           print('value: ' + str(value) + ' value_count: ' + str(value_count[value]) + ' gini_index: ' + str(gini_index))
    return gini_index, div_value


def SelectOptAttr(df):
    '''
    :param df: the pandas dataframe of the dataset
    :return: (the optimal attribute, the optimal divide value)
    '''
    min_gini = float('Inf')
    print(df)
    # for each attribute, compute info gain, and find the attribute which has maximum info gain
    for attr in df.columns[1:-1]:
        # compute info gain
        gini_index, div_value = GiniIndex(df, attr)
        print('attr: ' + str(attr) + ' gini_index: ' + str(gini_index) + ' div_value: ' + str(div_value))
        if gini_index < min_gini:
            min_gini = gini_index
            opt_attr = attr
            div_val = div_value
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
        node.label = max(label_count, key=label_count.get)
        return node
    # if attribute set A is NULL or the samples in D are the same value on attribute A,
    # set leaf node the class which has the most samples on set D
    if len(df.columns[1:-1]) == 0 or SameAttrValue(df):
        node.label = max(label_count, key=label_count.get)
        return node
    # select optimal divide attribute
    opt_attr, div_val = SelectOptAttr(df)
    node.attr = opt_attr
    if div_val == 0: # categoric variable
        # for each attribute value in a, generate a branch
        value_count = GetValueCount(df[opt_attr])
        for v in value_count:
            # get D_v by v
            df_v = df[df[opt_attr].isin([v])]
            # if the sample numbers of D_v is zero, set node the class which has the most samples in D
            if df_v.empty:
                node.attr_down[v] = Node(None, max(label_count, key=label_count.get), {})
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


def TreeToGraph(root, i, g):
    if root.attr == None:
        g_node_label = "Node:%d\n好瓜:%s" % (i, root.label)
    else:
        g_node_label = "Node:%d\n好瓜:%s\n属性:%s\n" % (i, root.label, root.attr)
    g_node = i
    g.add_node(graphviz.Node(g_node, label=g_node_label))

    for value in list(root.attr_down):
        i, g_child = TreeToGraph(root.attr_down[value], i+1, g)
        g.add_edge(graphviz.Edge(g_node, g_child, label=value))
    return i, g_node


def DrawPng(root, out_file):
    g = graphviz.Dot()
    TreeToGraph(root, 0, g)
    g.write("../data/tree.dot")


if __name__ == "__main__":
    df = loadDataSet("../data/watermelon_2.csv")
    index_train = [0, 1, 2, 5, 6, 9, 13, 14, 15, 16]
    df_train = df.iloc[index_train]
    root = TreeGenerate(df_train)
    DrawPng(root, "../data/desicion_tree.png")