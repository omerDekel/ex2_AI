from collections import defaultdict

import sys

from DTL import DTL, print_dt_to_file, read_test_file_DTL

"""
main class ,including knn functions and nb functions .
"""


def read_train_file_NB(file_info, attributes_to_values_dict):
    """
    read the train information and return accodingly the values to classify the test with .
    :param file_info: to read the the train data .
    :param attributes_to_values_dict: dictionary from attribute name to its set of values .
    :return: probability_atr_given_class dictionry, classes set of the possible classes, probability_class_arr
    dictionary of classes probabilities .
    """
    size = len(file_info)
    amount_atr_class = defaultdict(float)
    amount_class = defaultdict(float)
    classes_arr = []
    attributes_arr = []
    probability_class_arr = defaultdict(float)
    probability_atr_given_class = defaultdict(float)
    for line in file_info:
        classify = line[-1]
        # make dictionary class name to its amount
        amount_class[classify] += 1
        classes_arr.append(classify)
        # going through the attributes in line
        for atr in line[:-1]:
            # make dictionary atr given class name to its amount
            amount_atr_class[atr + ' ' + classify] += 1
            attributes_arr.append(atr)
    classes = set(classes_arr)
    attributes = set(attributes_arr)
    for c in classes_arr:
        # dictionary of classes probability
        probability_class_arr[c] = amount_class[c] / (size + 1)
        for a in attributes:
            # finding k for smoothing formula
            for name in attributes_to_values_dict:
                if a in attributes_to_values_dict[name]:
                    k = len(attributes_to_values_dict[name])
            # probablity with smoothing
            probability_atr_given_class[a + ' ' + c] = (amount_atr_class[a + ' ' + c] + 1) / (amount_class[c] + k)
    return probability_atr_given_class, classes, probability_class_arr


def read_test_file_NB(train_file, test_file, attributes_to_values_dict):
    """

    :param train_file: train file information to learn from it .
    :param test_file: test file information the nb learning check and classifies its examples .
    :return: classifications to examples in test by NB , and number of right classifications in test .
    """
    probability_atr_given_class, classes, probability_class_arr = read_train_file_NB(train_file,
                                                                                     attributes_to_values_dict)
    #
    C_NB = defaultdict(float)
    classifications = []
    right_predictions = 0
    # going through test examples .
    for line in test_file:
        # classifies by NB formula .
        for clas in classes:
            mult_prob = probability_class_arr[clas]
            for att in line[:-1]:
                mult_prob *= probability_atr_given_class[att + ' ' + clas]
            C_NB[clas] = mult_prob
        # saving the class with max probability
        max_prob = max(C_NB, key=lambda k: C_NB[k])
        classifications.append(max_prob)
        # if the classify is right
        if max_prob == line[-1]:
            right_predictions += 1
    return classifications, right_predictions


def hamming_distance(line_train, line_test):
    """
    :param line_train: example in train
    :param line_test: example in test
    :return: hamming distance between line_train to line_test
    """
    distance_counter = 0
    for train_attr, test_attr in zip(line_test[:-1], line_train[:-1]):
        if train_attr != test_attr:
            distance_counter += 1
    return distance_counter


def KNN(file_train, file_test, k):
    """

    :param file_train: train exaples .
    :param file_test: test examples .
    :param k: number of k nearest neighboors .
    :return: classifications answers got by knn learning
    """
    num_right_preditions = 0
    answers = []
    # for each example in test
    for line in file_test:
        # list of k nearest neighboors classes
        majority_classes_list = []
        hamming_dist_list = []
        # for example in test calculate the hamming distances from all train examples
        for j, train_line in enumerate(file_train):
            hamm = hamming_distance(train_line, line)
            hamming_dist_list.insert(j, hamm)
        # find k nearest neighboors and their classifications .
        for n in range(1, k + 1):
            # find minimum hamming distance
            mini_haming_dis = min(hamming_dist_list)
            # find minimum hamming distance index
            mini_haming_dis_index = hamming_dist_list.index(mini_haming_dis)
            hamming_dist_list.pop(mini_haming_dis_index)
            majority_classes_list.append(file_train[mini_haming_dis_index][-1])
            file_train.pop(mini_haming_dis_index)
        # find the majority class in k nearest neighboors.
        answer = max(majority_classes_list, key=majority_classes_list.count)
        # if prediction was right
        if answer == line[-1]:
            num_right_preditions += 1
        answers.append(answer)
    return answers, num_right_preditions


def get_attributes_to_values_dict(examples, attributes):
    """

    create dictionary between attributes and their optional values .
    :param examples: examples of file .
    :param attributes: attributes names .
    :return: dictionary between attributes and their optional values .
    """
    attributes_to_values_dict = defaultdict(set)
    for i, example in enumerate(examples):
        for j, val in enumerate(example):
            if j == (len(example) - 1):
                continue
            attributes_to_values_dict[attributes[j]].add(val)
    return attributes_to_values_dict


def read_file(file):
    """

    :param file: file of examples .
    :return: list of examples .
    """
    file_list = []
    attributes = []
    for count, l in enumerate(file):
        if count == 0:
            attributes = l.strip("\n").split('\t')
            attributes.pop(-1)
        else:
            file_list += [l.strip('\n').split('\t')]
    return attributes, file_list


def write_to_file(file_out,
                  #dt_out,
                    nb_out, knn_out,
                  #dtl_right_preditions,
                    knn_right_preditions, nb_right_preditions):
    """

    write the results to file .
    :param file_out: output file .
    :param dt_out: dtl classifications .
    :param nb_out: nb classifications .
    :param knn_out: knn classifications .
    :param dtl_right_preditions: number of right predictions by dtl .
    :param knn_right_preditions:number of right predictions by knn
    :param nb_right_preditions: number of right predictions by nb
    :return:
    """
    file_out.write("Num\tDT\tKNN\tnaiveBase\n")
    for num, (#dt,
              knn, nb) in enumerate(zip(
        #dt_out,
        knn_out, nb_out)):
        file_out.write(str(num) + "\t"

                       #+ dt
         + "\t" + knn + "\t" + nb + "\n")
    file_out.write("\t"
    #+ "%.2f" % (dtl_right_preditions)
     + "\t" + "%.2f" % (knn_right_preditions) + "\t" + "%.2f" % (
        nb_right_preditions))


def main():
    """
    main function .
    classifies the test examples by knn, nb and dtl
     and write the classifications and their accuracies and th decision tree to files.
    """
    file_test = open("test.txt", "r")
    file_train = open("train.txt", "r")
    # create list of train examples and list of its attributes .
    attributes, train_info = read_file(file_train)
    # create list of test examples and list of its attributes .
    attributes2, test_info = read_file(file_test)
    attributes_to_values_dict = get_attributes_to_values_dict(train_info, attributes)
    # create the dtl .
    #tree = DTL(train_info, attributes, attributes_to_values_dict)
    # print the tree to file
    file_output_tree = open("output_tree.txt", "w")
    #string_tree = print_dt_to_file(tree, attributes, "", 0)
    #file_output_tree.write(string_tree[1:]);
    # classify the test by dtl .
    #dtl_right_preditions, dtl_answers = read_test_file_DTL(attributes, tree, test_info)
    # classify the test by knn .
    knn_answers, knn_right_preditions = KNN(train_info, test_info, 5)
    # classify the test by nb .
    nb_answers, nb_right_preditions = read_test_file_NB(train_info, test_info, attributes_to_values_dict)
    file_out = open("output.txt", "w")
    size_of_test = len(test_info)
    # calculate the accuracies .
    #dtl_accuracy = dtl_right_preditions / float(size_of_test)
    knn_accuracy = knn_right_preditions / float(size_of_test)
    nb_accuracy = nb_right_preditions / float(size_of_test)
    # writing the output file .
    write_to_file(file_out, nb_answers, knn_answers     , knn_accuracy, nb_accuracy)


if __name__ == "__main__":
    main()

