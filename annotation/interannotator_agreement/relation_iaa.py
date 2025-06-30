import os
import glob
import itertools
import statistics
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def brat_to_dataframe(ann_path, annotator):
    """
    Convert Brat annotations to a pandas DataFrame.
    """
    files = glob.glob(os.path.join(ann_path, '*.ann'))
    annotations = []
    for file in files:
        with open(file, 'r') as f:
            doc_id = os.path.basename(f.name)
            text_line = [line for line in f]
            dict_entity = {line.strip().split('\t')[0]: line.strip().split('\t') for line in text_line if line.strip().split('\t')[0].startswith('T')}
            for line in text_line:
                line = line.strip().split('\t')
                if line[0].startswith('R'):
                    # Extract relation type and arguments
                    r_type, r_arg1, r_arg2 = line[1].split(' ')
                    r_arg1 = r_arg1.split(':')
                    r_arg2 = r_arg2.split(':')
                    entity_arg1 = dict_entity[r_arg1[1]][1].split(' ')
                    entity_arg2 = dict_entity[r_arg2[1]][1].split(' ')
                    annotations.append([doc_id, annotator, r_type, entity_arg1[0], entity_arg1[1], entity_arg1[2], entity_arg2[0], entity_arg2[1], entity_arg2[2]])
    df = pd.DataFrame(annotations, columns=['doc_id', 'annotator', 'relation_type', 'arg1', 'arg1_start', 'arg1_end', 'arg2', 'arg2_start', 'arg2_end'])
    return df

def get_relation_pairs(df, annotators):
    """
    Get all pairs of relations annotated by the specified annotators.
    """
    dict_doc = {}
    for doc_id in df['doc_id'].unique():
        for rel_type in df['relation_type'].unique():

            # filtering dataframe on doc_id and relation_type
            doc_df = df[(df['doc_id'] == doc_id) & (df['relation_type'] == rel_type)]
            if not doc_df.empty:

                # filtering on annotators
                a1 = annotators[0]
                a2 = annotators[1]
                a1_df = doc_df[doc_df['annotator'] == a1]
                a2_df = doc_df[doc_df['annotator'] == a2]

                args1 = doc_df['arg1'].unique()
                args2 = doc_df['arg2'].unique()

                a1_vec = []
                a2_vec = []

                if not a1_df.empty and not a2_df.empty:

                    for arg1, arg2 in itertools.product(args1, args2):
                        a1_args_df = a1_df[(a1_df['arg1'] == arg1) & (a1_df['arg2'] == arg2)]
                        a2_args_df = a2_df[(a2_df['arg1'] == arg1) & (a2_df['arg2'] == arg2)]
                        if len(a1_args_df) == len(a2_args_df):
                            a1_vec += [1]*len(a1_args_df)
                            a2_vec += [1]*len(a1_args_df)
                        elif len(a1_args_df) > len(a2_args_df):
                            a1_vec += [1]*len(a1_args_df)
                            a2_vec += [1]*len(a2_args_df)+[0]*(len(a1_args_df)-len(a2_args_df))
                        else:
                            a1_vec += [1]*len(a1_args_df)+[0]*(len(a2_args_df)-len(a1_args_df))
                            a2_vec += [1]*len(a2_args_df)
                else: # one of the annotators did not annotate this relation rel_type
                    if a1_df.empty:
                        a1_vec += [0]*len(a2_df)
                        a2_vec += [1]*len(a2_df)
                    elif a2_df.empty:
                        a1_vec += [1]*len(a1_df)
                        a2_vec += [0]*len(a1_df)
                if doc_id in dict_doc:
                    dict_doc[doc_id][rel_type] = (a1_vec, a2_vec)
                else:
                    dict_doc[doc_id] = {rel_type: (a1_vec, a2_vec)}
    return dict_doc

def compute_f1(list1, list2):
    """
    Compute F measure for inter-annotator agreement.
    """
    precision, recall, f1_score, _ = precision_recall_fscore_support(list1, list2, average='binary')
    return f1_score

if __name__ == '__main__':
    # Specify the path to the Brat annotations directory
    ann_path_1 = './bratiia_synthetic/annotation_synthetic_a1'
    ann_path_2 = './bratiia_synthetic/annotation_synthetic_a2'
    ann_path_3 = './bratiia_synthetic/annotation_synthetic_a3'

    # Specify the annotators to include in the analysis
    annotators = ['a1', 'a2', 'a3']

    list_relations = ['History', 'Status', 'Frequency', 'Amount', 'Type', 'Duration']
    # Convert Brat annotations to a pandas DataFrame
    df_a1 = brat_to_dataframe(ann_path_1, 'a1')
    df_a2 = brat_to_dataframe(ann_path_2, 'a2')
    df_a3 = brat_to_dataframe(ann_path_3, 'a3')
    frames = [df_a1, df_a2, df_a3]
    result = pd.concat(frames)
    pairs = get_relation_pairs(result, annotators)
    print(pairs)

    f1_values = []

    for relation in list_relations:

        vec_a1 = []
        vec_a2 = []

        for doc_id in pairs:
            if relation in pairs[doc_id]:
                vec_a1 += pairs[doc_id][relation][0]
                vec_a2 += pairs[doc_id][relation][1]

        print('Relation :', relation)
        print(len(vec_a1))
        print(len(vec_a2))
        print(compute_f1(vec_a1, vec_a2))
        f1_values.append(compute_f1(vec_a1, vec_a2))

    print("Average:", statistics.mean(f1_values))

