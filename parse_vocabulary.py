#!/usr/bin/python
import cPickle, getopt, sys, time, re
import datetime, os;

import scipy.io;
import nltk;
import numpy;
import optparse;
import collections
import operator

def main(data_path, vocab_path):
    input_stream = open(data_path, "r");
    term_frequency = {}
    doc_frequency = {}
    for line in input_stream:
        line = line.strip();
        tokens = line.split("\t")[0].split(" ");
        for token in set(tokens):
            if token not in doc_frequency:
                doc_frequency[token] = 0;
            if token not in term_frequency:
                term_frequency[token] = 0;

        for token in set(tokens):
            doc_frequency[token] += 1;
        for token in tokens:
            term_frequency[token] += 1;

    assert len(term_frequency)==len(doc_frequency)

    output_stream = open(vocab_path, 'w');
    for token, dummy in sorted(term_frequency.items(), key=operator.itemgetter(1), reverse=True):
        output_stream.write("%s\t%d\t%d\n" % (token, term_frequency[token], doc_frequency[token]));

if __name__ == '__main__':
    data_path = sys.argv[1]
    vocab_path = sys.argv[2]
    main(data_path, vocab_path);
