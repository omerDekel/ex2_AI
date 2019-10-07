from collections import defaultdict

import sys

import math

"""
DTL class responsible for creating dtl and classify the test examples accordingly , and its file printing .
"""


def mode(examples):
    """

    :param examples: examples .
    :return: return the most common class in  examples .
    """
    amount_classes_map = defaultdict(int)
    for example in examples:
        amount_classes_map[example[-1]] += 1
    return max(amount_classes_map, key=lambda key: amount_classes_map[key])


def gain(examples, attribute_values):
    """

    :param examples: examples .
    :param attribute_values: specific attribute's values .
    :return: the gain according the examples that have the attribute's values .
    """
    entropy_d = entropy(examples)
    for atr in attribute_values:
        # get elements of examples with atr value .
        elements_example = get_elements_examples(atr, examples)
        probability = len(elements_example) / float(len(examples))
        entropy_d -= entropy(elements_example) * probability
    return entropy_d


def entropy(examples):
    """
    implements the entropy formula .
    :param examples: examples .
    :return: entropy according the examples .
    """
    amount = defaultdict(int)
    size = len(examples)
    classes = []
    sum_entropy = 0
    for example in examples:
        clas = example[-1]
        classes.append(clas)
        amount[clas] += 1
    classes = set(classes)
    for c in classes:
        probability_decision_given_atr = amount[c] / float(size)
        sum_entropy -= probability_decision_given_atr * math.log(probability_decision_given_atr, 2)
    return sum_entropy


def delete_atr(attributes, atr):
    """
    :param attributes: list of attributes.
    :param atr:  specific attribute .
    :return:  new list of attributes without attribute atr .
    """
    new_atr = []
    for at in attributes:
        if at != atr:
            new_atr.append(at)
    return new_atr


def get_elements_examples(attr, examples):
    """

    :param attr: specific attribute value.
    :param examples: examples list .
    :return: get elements of examples with atr value .
    """
    examples_with_attr = []
    for i, example in enumerate(examples):
        for at in example:
            if at == attr:
                examples_with_attr.append(example)
    return examples_with_attr


def choose_attribute(attributes, examples, attribute_vals):
    """

    :param attributes:  attributes list .
    :param examples: examples list .
    :param attribute_vals: dictionary from attributes to set of their values.
    :return:  the best attribute according to its information gain.
    """
    # dictionary between attribute and its gain .
    gain_atr = defaultdict(float)
    for i, attribute in enumerate(attributes):
        gain_atr[attribute] = gain(examples, attribute_vals[attribute])
    # return the the attribute with the maximum gain .
    return max(gain_atr, key=lambda key: gain_atr[key])


def DTL(examples, attributes, attributes_to_values_dict, default=None):
    """

    :param examples: list of examples.
    :param attributes: list of attributes .
    :param attributes_to_values_dict: dictionary between attributes to set of their values.
    :param default: default return value .
    :return: decision tree learning .
    """
    # if examples is empty
    if not examples:
        return default
    # if all examples have the same classification
    classes = []
    for i, exa in enumerate(examples):
        classes.insert(i, exa[-1])
    classes_set = set(classes)
    if (len(classes_set) == 1):
        return examples[0][-1]
    # if attributes is empty
    if not attributes:
        return mode(examples)
    best_attribute = choose_attribute(attributes, examples, attributes_to_values_dict)
    # new  decision tree with root test best
    decision_tree = {best_attribute: {}}
    # values of best_attribute
    values_best = attributes_to_values_dict[best_attribute]
    attributes_without_best = delete_atr(attributes, best_attribute)
    # for each value of best_attribute
    for vi in values_best:
        # elements of examples with best_attribute = vi
        examples_vi = get_elements_examples(vi, examples)
        sub_decision_tree = DTL(examples_vi, attributes_without_best, attributes_to_values_dict, mode(examples))
        # add a branch to label vi and subtree sub_decision_tree
        decision_tree[best_attribute][vi] = sub_decision_tree
    return decision_tree


def read_test_file_DTL(attributes, decision_tree, test_info):
    """
    classify the test examples according to decision tree learning .
    :param attributes: list of attributes .
    :param decision_tree: decision_tree we classify by him .
    :param test_info: test examples .
    :return: list of classifications to test examples and number of right predictions .
    """
    # answers of classifications to the examples .
    answers = []
    right_predictions = 0
    # for each example in test .
    for line in test_info:
        tree_i = decision_tree
        decision = get_Decision(attributes, tree_i, line)
        # if the classification was right .
        if decision == line[-1]:
            right_predictions += 1
        answers.append(decision)
    return right_predictions, answers


def get_Decision(attributes, tree_i, line):
    """

    :param attributes: list of attributes .
    :param tree_i: decision tree we get a decision by him .
    :param line:  line of example to classify
    :return: the classify of line by the decision tree .
    """
    # the decision cant be one of these types
    while isinstance(tree_i, list) or isinstance(tree_i, dict):
        # find the primary attribute key in the dictionary
        for attri in attributes:
            if attri in tree_i:
                # updating the tree
                tree_i = tree_i[attri]
                break
        # going through the values attributes in the example .
        for i, value_atr in enumerate(line):
            if i == len(line) - 1:
                continue
            # find the corresponding attribute value in the line and updating the tree according it
            if value_atr in tree_i:
                tree_i = tree_i[value_atr]
                break
    # we got the decision in the tree
    return tree_i


def print_dt_to_file(tree, attributes, pre_node, count, tree_string=""):
    """
    print the decision tree to file .
    :param tree: decision tree to print .
    :param attributes: attributes list .
    :param pre_node: previous node of the current node .
    :param count: iterations counter .
    :return: dtl as string .
    """
    
    # if the tree isn't leaf(decision string)
    if type(tree) != str:
        # going through the tree nodes
        for node in sorted(tree):
            # if the previous node is attribute
            if pre_node in attributes:
                    tree_string += '\n'
                    # if it's not the first iteration .
                    if count > 2:
                        if type(tree[node]) == str:
                            tree_string += (('\t' * (count - 2)) + '|' + pre_node + '=' + node+":" + tree[node])
                        else:
                            tree_string += (('\t' * (count - 2)) + '|' + pre_node + '=' + node)
                    else:
                        if type(tree[node]) == str:
                            tree_string += (('\t' * (count - 2)) + pre_node + '=' + node+":" + tree[node])
                        else:
                            tree_string += (('\t' * (count - 2))  + pre_node + '=' + node)

            # if the node is tree (dictionary) call the function recursively
            if isinstance(tree, dict):
                tree_string += print_dt_to_file(tree[node], attributes, node, count + 1)
    return tree_string