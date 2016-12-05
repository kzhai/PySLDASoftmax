#!/usr/bin/python
import cPickle, getopt, sys, time, re
import datetime, os;

import scipy.io;
import nltk;
import numpy;
import optparse;

def main(input_directory, output_directory):
    input_train_path = os.path.join(input_directory, "train.dat")
    output_train_path = os.path.join(output_directory, "train.dat")
    parse_image(input_train_path, output_train_path);

    input_test_path = os.path.join(input_directory, "test.dat")
    output_test_path = os.path.join(output_directory, "test.dat")
    parse_image(input_test_path, output_test_path);

def parse_image(input_path, output_path):
    input_stream = open(input_path, 'r');
    output_stream = open(output_path, 'w');
    for line in input_stream:
        line = line.strip();
        tokens = [];

        fields = line.split("\t");
        label = int(fields[1]);
        token_count_pairs = fields[0].split(" ");
        number_of_tokens = int(token_count_pairs[0]);
        for type_count_pair in token_count_pairs[1:]:
            type_count_pair_split = type_count_pair.split(":");
            token_index = int(type_count_pair_split[0]);
            token_count = int(type_count_pair_split[1]);

            for x in xrange(token_count):
                tokens.append("%i" % token_index);

        output_stream.write("%s\t%i\n" % (" ".join(tokens), label));

if __name__ == '__main__':
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    main(input_directory, output_directory);
