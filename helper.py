import time
import sys
sys.path.insert(1, './mmd')
from mmd import diagnoser
from scipy import stats as st
import numpy
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

def get_top_f1_rules(MD_result):
    new_rule_states = []
    rule_stats = MD_result.get_rule_stats()
    for idx, rule in enumerate(rule_stats):
        precision = rule['precision']
        recall = rule['recall']
        f1 = 2 * precision * recall/(precision + recall)
        rule['F1'] = f1
        new_rule_states.append(rule)

    top_f1_rules = []

    for rule in rule_stats:
        top_f1_rules.append([rule['F1'], rule['rule']]) #, rule['precision'], rule['recall']
    top_f1_rules.sort(reverse=True, key=lambda x: x[0])

    return top_f1_rules


def get_relevent_attributs_target(X_test):
    
    Target = diagnoser.Target('mispredict', value = True)
    
    target_name, target_value = Target
    relevant_attributes = X_test.dtypes.to_dict()
    for attribute, dtype in relevant_attributes.items():
        if dtype == "float64":
            relevant_attributes[attribute] = 'C'
        elif dtype == "int64":
            relevant_attributes[attribute] = 'I'
        else:
            relevant_attributes[attribute] = 'D'
    return (relevant_attributes, Target)

def get_MMD_results(X_test, relevant_attributes, Target, settings=diagnoser.Settings):
        
    start = time.time()
    original_MMD_result = diagnoser.discover(X_test, Target, relevant_attributes, settings)
    end = time.time()
    print("Original Rule")
    original_MMD_result.print()
    print("time:", end-start)
    print("Feature Number", len(relevant_attributes))
    
    top_f1_rules_original = get_top_f1_rules(original_MMD_result)
    
    return (top_f1_rules_original, end - start, len(relevant_attributes))

def get_biased_features(X_test, relevant_attributes):
    
    features = X_test.columns.to_list()
    features.remove('mispredict')

    correct_values = X_test[X_test['mispredict'] == False]
    wrong_values = X_test[X_test['mispredict'] == True]
    
    biased_attributes = {}
    for feature in features:
        # get correct/wrong values
        feature_correct_values = correct_values[feature].to_list()
        feature_wrong_values = wrong_values[feature].to_list()
        # Get T-test result for each feature
        ttest_result = st.ttest_ind(a=numpy.array(feature_correct_values),
                                    b=numpy.array(feature_wrong_values),
                                    equal_var=True)
        
        if ttest_result.pvalue < 0.05:
            # print(feature, 'p-Value', ttest_result.pvalue)
            biased_attributes[feature] = relevant_attributes[feature]

    return biased_attributes

def get_BGMD_results(X_test, relevant_attributes, Target, settings=diagnoser.Settings):

    biased_attributes = get_biased_features(X_test, relevant_attributes)

    

    start = time.time()
    biased_MMD_result = diagnoser.discover(X_test, Target, biased_attributes, settings)
    end = time.time()
    print()
    print("###############################")
    print()
    print("BGMD Rule")
    # biased_attributes.print()
    biased_MMD_result.print()
    print("time:", end-start)
    print("Feature Number", len(biased_attributes))
    
    top_f1_rules_bias = get_top_f1_rules(biased_MMD_result)

    return (top_f1_rules_bias, end-start, len(biased_attributes))


def generateTrain_data_Weights(BGMD_rules, X_train, upweight_value=2, default_weight=1):
    rules = []
    for rule in BGMD_rules:
        if '&' in rule[1]:
            sub_rules = rule[1].split('&')
            for sub_rule in sub_rules:
                temp = []
                if '>' in sub_rule:
                    splited = sub_rule.split('>')
                    temp.append((splited[0], '>', splited[1]))
                elif '<=' in sub_rule:
                    splited = sub_rule.split('<=')
                    temp.append((splited[0], '<=', splited[1]))
            rules.append(temp)
        else:
            if '>' in rule[1]:
                splited = rule[1].split('>')
                rules.append((splited[0], '>', splited[1]))
            elif '<=' in rule[1]:
                splited = rule[1].split('<=')
                rules.append((splited[0], '<=', splited[1]))
    
    weights = []
    indexes_in_misprediction_area = []

    for index, row in X_train.iterrows():
        add_weight = False
        # add upweight_value if sample meet any rule
        for rule in rules:
            if type(rule) == tuple and not add_weight:
                rule_col_name = rule[0].strip()
                operator = rule[1].strip()
                rule_value = float(rule[2])
                if operator == '>' and row[rule_col_name] > rule_value and not add_weight:
                    weights.append(upweight_value)
                    add_weight = True
                elif operator == '<=' and row[rule_col_name] <= rule_value and not add_weight:
                    weights.append(upweight_value)
                    add_weight = True

            elif type(rule) == list and not add_weight:
                num_sub_rule = len(rule)
                mathced = 0 
                for sub_rule in rule:
                    rule_col_name = sub_rule[0].strip()
                    operator = sub_rule[1].strip()
                    rule_value = float(sub_rule[2])
                    
                    if operator == '>' and row[rule_col_name] > rule_value and not add_weight:
                        mathced += 1
                    elif operator == '<=' and row[rule_col_name] <= rule_value and not add_weight:
                        mathced += 1
                if mathced == num_sub_rule:
                    weights.append(upweight_value)
                    add_weight = True

        # add default_weight if not match any rule 
        if not add_weight:
            weights.append(default_weight)
            
    return weights


def get_test_data_in_misprediction_areas(BGMD_rules, X_test):
    rules = []
    for rule in BGMD_rules:
        if '&' in rule[1]:
            sub_rules = rule[1].split('&')
            for sub_rule in sub_rules:
                temp = []
                if '>' in sub_rule:
                    splited = sub_rule.split('>')
                    temp.append((splited[0], '>', splited[1]))
                elif '<=' in sub_rule:
                    splited = sub_rule.split('<=')
                    temp.append((splited[0], '<=', splited[1]))
            rules.append(temp)
        else:
            if '>' in rule[1]:
                splited = rule[1].split('>')
                rules.append((splited[0], '>', splited[1]))
            elif '<=' in rule[1]:
                splited = rule[1].split('<=')
                rules.append((splited[0], '<=', splited[1]))
    
    indexes_in_misprediction_area = []

    for index, row in X_test.iterrows():
        add_index = False
        # add index if sample in a rule
        for rule in rules:
            if type(rule) == tuple:
                rule_col_name = rule[0].strip()
                operator = rule[1].strip()
                rule_value = float(rule[2])
                
                if operator == '>' and row[rule_col_name] > rule_value and not add_index:
                    indexes_in_misprediction_area.append(index)
                    add_index = True
                elif operator == '<=' and row[rule_col_name] <= rule_value and not add_index:
                    indexes_in_misprediction_area.append(index)
                    add_index = True
            elif type(rule) == list:
                num_sub_rule = len(rule)
                matched = 0 
                for sub_rule in rule:
                    rule_col_name = sub_rule[0].strip()
                    operator = sub_rule[1].strip()
                    rule_value = float(sub_rule[2])
                    
                    if operator == '>' and row[rule_col_name] > rule_value and not add_index:
                        matched += 1
                    elif operator == '<=' and row[rule_col_name] <= rule_value and not add_index:
                        matched += 1
                if matched == num_sub_rule:
                    indexes_in_misprediction_area.append(index)
                    add_index = True
            
    return indexes_in_misprediction_area

def get_performance(X_test, y_test, y_pred):
    # print('Performance on all data')
    total_result = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    # print('Precision:', total_result[0])
    # print('Recall:', total_result[1])
    # print('F1 Score:', total_result[2])
    return total_result

def get_conflict_commit_performance(X_test, y_test, y_pred):
    # print('Performance on all data')
    total_result = precision_recall_fscore_support(y_test, y_pred, labels=[1])
    # print('Precision:', total_result[0][0])
    # print('Recall:', total_result[1][0])
    # print('F1 Score:', total_result[2][0])
    return total_result
    # test_data = pd.concat([X_test, y_test], axis=1, join='inner')
    # test_data.loc[:,"pred"] = y_pred
    # conflict_only = test_data.loc[test_data["is_conflict"] == 1]
    
    # print()
    # print('Performance on merge conflict data')
    # conflict_only_result = precision_recall_fscore_support(conflict_only['is_conflict'], conflict_only['pred'], average='weighted')
    # print('Precision:', conflict_only_result[0])
    # print('Recall:', conflict_only_result[1])
    # print('F1 Score:', conflict_only_result[2])

def get_mispredicted_region_test(X_test, y_test, y_pred_default, y_pred_SMOTE, y_pred_MAPS, BGMD_rules, ylabel='OOSLA'):

    final_result = pd.concat([X_test, y_test], axis=1, join='inner')
    final_result['y_pred_default'] = y_pred_default
    final_result['y_pred_SMOTE'] = y_pred_SMOTE
    final_result['y_pred_MAPS'] = y_pred_MAPS

    indexes_in_misprediction_area = get_test_data_in_misprediction_areas(BGMD_rules, X_test)
    
    y_actual_MD = []
    y_predict_default_MD = []
    y_pred_SMOTE_MD = []
    y_pred_MAPS_MD = []

    for index in indexes_in_misprediction_area:
        y_actual_MD.append(final_result.loc[index][ylabel])
        y_predict_default_MD.append(final_result.loc[index]['y_pred_default'])
        y_pred_SMOTE_MD.append(final_result.loc[index]['y_pred_SMOTE'])
        y_pred_MAPS_MD.append(final_result.loc[index]['y_pred_MAPS'])


    print('y_actual_MD:', len(y_actual_MD))
    print('y_predict_default_MD:', len(y_predict_default_MD))
    print('y_pred_SMOTE_MD:', len(y_pred_SMOTE_MD))
    print('y_pred_MAPS_MD:', len(y_pred_MAPS_MD))
    print()

    default_MD_metric = precision_recall_fscore_support(y_actual_MD, y_predict_default_MD, average='weighted')
    SMOTE_MD_metric = precision_recall_fscore_support(y_actual_MD, y_pred_SMOTE_MD, average='weighted')
    MAPS_MD_metric = precision_recall_fscore_support(y_actual_MD, y_pred_MAPS_MD, average='weighted')
    print("Default:", default_MD_metric)
    print("SMOTE:", SMOTE_MD_metric)
    print("MAPS:", MAPS_MD_metric)
    
    return (default_MD_metric, SMOTE_MD_metric, MAPS_MD_metric)

def get_merge_coflict_mispredicted_region_test(X_test, y_test, y_pred_default, y_pred_SMOTE, y_pred_MAPS, BGMD_rules, ylabel='is_conflict'):

    final_result = pd.concat([X_test, y_test], axis=1, join='inner')
    final_result['y_pred_default'] = y_pred_default
    final_result['y_pred_SMOTE'] = y_pred_SMOTE
    final_result['y_pred_MAPS'] = y_pred_MAPS

    indexes_in_misprediction_area = get_test_data_in_misprediction_areas(BGMD_rules, X_test)
    
    y_actual_MD = []
    y_predict_default_MD = []
    y_pred_SMOTE_MD = []
    y_pred_MAPS_MD = []

    for index in indexes_in_misprediction_area:
        y_actual_MD.append(final_result.loc[index][ylabel])
        y_predict_default_MD.append(final_result.loc[index]['y_pred_default'])
        y_pred_SMOTE_MD.append(final_result.loc[index]['y_pred_SMOTE'])
        y_pred_MAPS_MD.append(final_result.loc[index]['y_pred_MAPS'])


    # print('y_actual_MD:', len(y_actual_MD))
    # print('y_predict_default_MD:', len(y_predict_default_MD))
    # print('y_pred_SMOTE_MD:', len(y_pred_SMOTE_MD))
    # print('y_pred_MAPS_MD:', len(y_pred_MAPS_MD))
    # print()

    default_MD_metric = precision_recall_fscore_support(y_actual_MD, y_predict_default_MD, labels=[1])
    SMOTE_MD_metric = precision_recall_fscore_support(y_actual_MD, y_pred_SMOTE_MD, labels=[1])
    print("Default:", default_MD_metric)
    print("SMOTE:", SMOTE_MD_metric)
    
    # return (default_MD_metric, SMOTE_MD_metric, MAPS_MD_metric)

def generate_JTT_Weights(y_val, y_pred, weight=2, default_weight=1):
    weights = []
    for idx in range(len(y_val)):
        # add weight_value if mispredicted
        if y_val[idx] == y_pred[idx]:
            weights.append(weight)
        # add default_weight if not mispredicted 
        else:
            weights.append(default_weight)
            
    return numpy.array(weights)