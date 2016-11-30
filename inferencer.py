"""
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import time
import numpy
import scipy
import nltk;
import sys;

"""
This is a python implementation of vanilla lda, based on a lda approach of variational inference and gibbs sampling, with hyper parameter updating.
It supports asymmetric Dirichlet prior over the topic simplex.
"""

def compute_dirichlet_expectation(dirichlet_parameter):
    if (len(dirichlet_parameter.shape) == 1):
        return scipy.special.psi(dirichlet_parameter) - scipy.special.psi(numpy.sum(dirichlet_parameter))
    return scipy.special.psi(dirichlet_parameter) - scipy.special.psi(numpy.sum(dirichlet_parameter, 1))[:, numpy.newaxis]

'''
def parse_vocabulary(vocab):
    type_to_index = {};
    index_to_type = {};
    for word in set(vocab):
        index_to_type[len(index_to_type)] = word;
        type_to_index[word] = len(type_to_index);
        
    return type_to_index, index_to_type;
'''

class Inferencer():
    """
    """
    def __init__(self,
                 # local_parameter_iterations=50,
                 hyper_parameter_optimize_interval=10,
                 padding_default_label=False,
                 ):
        
        self._hyper_parameter_optimize_interval = hyper_parameter_optimize_interval;
        assert(self._hyper_parameter_optimize_interval > 0);
        
        self._padding_default_label = padding_default_label;
        # self._local_parameter_iterations = local_parameter_iterations
        # assert(self._local_maximum_iteration>0)

    """
    """
    def _initialize(self, vocab, labels, number_of_topics, alpha_alpha, alpha_beta):
        self.parse_vocabulary(vocab);
        self.parse_label(labels)
        
        # initialize the size of the vocabulary, i.e. total number of distinct tokens.
        self._number_of_types = len(self._type_to_index)
        self._number_of_labels = len(self._label_to_index)
        
        self._counter = 0;

        # initialize the total number of topics.
        self._number_of_topics = number_of_topics
        
        # initialize a K-dimensional vector, valued at 1/K.
        self._alpha_alpha = numpy.zeros(self._number_of_topics) + alpha_alpha;
        self._alpha_beta = numpy.zeros(self._number_of_types) + alpha_beta;
        # self._alpha_eta = numpy.zeros(self._number_of_types) + alpha_eta;
    
    def parse_vocabulary(self, vocab):
        self._type_to_index = {};
        self._index_to_type = {};
        for word in set(vocab):
            self._index_to_type[len(self._index_to_type)] = word;
            self._type_to_index[word] = len(self._type_to_index);
            
        self._vocab = self._type_to_index.keys();
    
    def parse_label(self, labels):
        if self._padding_default_label:
            self._label_to_index = {"default_label":0};
            self._index_to_label = {0:"default_label"};
        else:
            self._label_to_index = {};
            self._index_to_label = {};

        for label in set(labels):
            self._label_to_index[label] = len(self._label_to_index);
            self._index_to_label[len(self._index_to_label)] = label;
            
    def parse_data(self, corpus):
        doc_count = 0
        
        word_idss = [];
        label_idss = []
        
        for document_line in corpus:
            fields = document_line.split("\t");
            document = fields[0];
            word_ids = [];
            for token in document.split():
                if token not in self._type_to_index:
                    continue;
                
                type_id = self._type_to_index[token];
                word_ids.append(type_id);
            
            if len(word_ids) == 0:
                sys.stderr.write("warning: document collapsed during parsing");
                continue;
            
            word_idss.append(word_ids);
            
            if len(fields) > 1:
                labels = fields[1].split();
                label_ids = []
                if self._padding_default_label:
                    label_ids.append(0);
                for label in labels:
                    if label not in self._label_to_index:
                        sys.stderr.write("warning: label not found");
                        continue;
                    label_ids.append(self._label_to_index[label]);
                label_idss.append(label_ids)
            
            doc_count += 1
            if doc_count % 10000 == 0:
                print "successfully parse %d documents..." % doc_count;
        
        print "successfully parse %d documents..." % (doc_count);
        
        if len(label_idss) == 0:
            return word_idss, None;
        else:
            return word_idss, label_idss;

    """
    """
    def learning(self):
        raise NotImplementedError;
    
    """
    """
    def inference(self):
        raise NotImplementedError;
    
    """
    """
    def export_beta(self, exp_beta_path, top_display=-1):
        raise NotImplementedError;
    
if __name__ == "__main__":
    raise NotImplementedError;
