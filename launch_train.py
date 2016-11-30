#!/usr/bin/python
import cPickle, getopt, sys, time, re
import datetime, os;

import scipy.io;
import nltk;
import numpy;
import optparse;

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        output_directory=None,
                        # dictionary=None,
                        
                        # parameter set 2
                        training_iterations=-1,
                        snapshot_interval=10,
                        number_of_topics=-1,

                        # parameter set 3
                        alpha_alpha=-1,
                        alpha_beta=-1,
                        alpha_eta=-1,
                        eta_l2_lambda=1.0,
                        #alpha_sigma=1.0,
                        
                        # parameter set 4
                        # disable_alpha_theta_update=False,
                        inference_mode=2,
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]");
    # parser.add_option("--corpus_name", type="string", dest="corpus_name",
                      # help="the corpus name [None]")
    # parser.add_option("--dictionary", type="string", dest="dictionary",
                      # help="the dictionary file [None]")
    
    # parameter set 2
    parser.add_option("--number_of_topics", type="int", dest="number_of_topics",
                      help="total number of topics [-1]");
    parser.add_option("--training_iterations", type="int", dest="training_iterations",
                      help="total number of iterations [-1]");
    parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
                      help="snapshot interval [10]");                      
                      
    # parameter set 3
    parser.add_option("--alpha_alpha", type="float", dest="alpha_alpha",
                      help="hyper-parameter for Dirichlet distribution of topics [1.0/number_of_topics]")
    parser.add_option("--alpha_beta", type="float", dest="alpha_beta",
                      help="hyper-parameter for Dirichlet distribution of vocabulary [1.0/number_of_types]")
    parser.add_option("--alpha_eta", type="float", dest="alpha_eta",
                      help="hyper-parameter for Gaussian response [1.0/number_of_labels]")
    parser.add_option("--eta_l2_lambda", type="float", dest="l2_lambda",
                      help="l2 regularizer coefficient for eta [1.0]")
    #parser.add_option("--alpha_sigma", type="float", dest="alpha_sigma",
                      #help="hyper-parameter for Gaussian response [1.0]")
    
    # parameter set 4
    # parser.add_option("--disable_alpha_theta_update", action="store_true", dest="disable_alpha_theta_update",
                      # help="disable alpha (hyper-parameter for Dirichlet distribution of topics) update");
    parser.add_option("--inference_mode", type="int", dest="inference_mode",
                      help="inference mode [ " + 
                            "0 (default): hybrid inference, " + 
                            "1: monte carlo, " + 
                            "2: variational bayes " + 
                            "]");
    # parser.add_option("--inference_mode", action="store_true", dest="inference_mode",
    #                  help="run latent Dirichlet allocation in lda mode");

    (options, args) = parser.parse_args();
    return options;

def main():
    options = parse_args();
    
    # parameter set 2
    assert(options.number_of_topics > 0);
    number_of_topics = options.number_of_topics;
    assert(options.training_iterations > 0);
    training_iterations = options.training_iterations;
    assert(options.snapshot_interval > 0);
    if options.snapshot_interval > 0:
        snapshot_interval = options.snapshot_interval;
    
    # parameter set 4
    # disable_alpha_theta_update = options.disable_alpha_theta_update;
    inference_mode = options.inference_mode;
    
    # parameter set 1
    # assert(options.corpus_name!=None);
    assert(options.input_directory != None);
    assert(options.output_directory != None);
    
    input_directory = options.input_directory;
    input_directory = input_directory.rstrip("/");
    corpus_name = os.path.basename(input_directory);
    
    output_directory = options.output_directory;
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    output_directory = os.path.join(output_directory, corpus_name);
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    
    # Document
    train_docs_path = os.path.join(input_directory, 'train.dat')
    input_doc_stream = open(train_docs_path, 'r');
    train_docs = [];
    for line in input_doc_stream:
        train_docs.append(line.strip().lower());
    print "successfully load all training docs from %s..." % (os.path.abspath(train_docs_path));
    
    # Vocabulary
    vocabulary_path = os.path.join(input_directory, 'voc.dat');
    input_voc_stream = open(vocabulary_path, 'r');
    vocab = [];
    for line in input_voc_stream:
        vocab.append(line.strip().lower().split()[0]);
    vocab = list(set(vocab));
    print "successfully load all the words from %s..." % (os.path.abspath(vocabulary_path));
    
    # Labels
    labels_path = os.path.join(input_directory, 'label.dat');
    input_labels_stream = open(labels_path, 'r');
    labels = [];
    for line in input_labels_stream:
        labels.append(line.strip().lower().split()[0]);
    labels = list(set(labels));
    print "successfully load all the labels from %s..." % (os.path.abspath(labels_path));
    
    # parameter set 3
    alpha_alpha = 1.0 / number_of_topics;
    if options.alpha_alpha > 0:
        alpha_alpha = options.alpha_alpha;
    alpha_beta = options.alpha_beta;
    if alpha_beta <= 0:
        alpha_beta = 1.0 / len(vocab);

    #alpha_eta = 1.0 / len(labels);
    #if options.alpha_eta > 0:
        #alpha_eta = options.alpha_eta;

    eta_l2_lambda = options.eta_l2_lambda;
    assert eta_l2_lambda >= 0;

    #alpha_sigma = options.alpha_sigma;
    #assert alpha_sigma > 0
    
    # create output directory
    now = datetime.datetime.now();
    suffix = now.strftime("%y%m%d-%H%M%S") + "";
    suffix += "-%s" % ("slda");
    suffix += "-I%d" % (training_iterations);
    suffix += "-S%d" % (snapshot_interval);
    suffix += "-K%d" % (number_of_topics);
    suffix += "-aa%f" % (alpha_alpha);
    suffix += "-ab%f" % (alpha_beta);
    #suffix += "-ae%f" % (alpha_eta);
    #suffix += "-l2l%f" % (eta_l2_lambda);
    #suffix += "-as%f" % (alpha_sigma);
    suffix += "-im%d" % (inference_mode);
    # suffix += "-%s" % (resample_topics);
    # suffix += "-%s" % (hash_oov_words);
    suffix += "/";
    
    output_directory = os.path.join(output_directory, suffix);
    os.mkdir(os.path.abspath(output_directory));
        
    # store all the options to a file
    options_output_file = open(output_directory + "option.txt", 'w');
    # parameter set 1
    options_output_file.write("input_directory=" + input_directory + "\n");
    options_output_file.write("corpus_name=" + corpus_name + "\n");
    # options_output_file.write("vocabulary_path=" + str(dict_file) + "\n");
    # parameter set 2
    options_output_file.write("training_iterations=%d\n" % (training_iterations));
    options_output_file.write("snapshot_interval=" + str(snapshot_interval) + "\n");
    options_output_file.write("number_of_topics=" + str(number_of_topics) + "\n");
    # parameter set 3
    options_output_file.write("alpha_alpha=" + str(alpha_alpha) + "\n");
    options_output_file.write("alpha_beta=" + str(alpha_beta) + "\n");
    #options_output_file.write("alpha_eta=" + str(alpha_eta) + "\n");
    options_output_file.write("eta_l2_lambda=" + str(eta_l2_lambda) + "\n");
    #options_output_file.write("alpha_sigma=" + str(alpha_sigma) + "\n");
    # parameter set 4
    options_output_file.write("inference_mode=%d\n" % (inference_mode));
    options_output_file.close()
    
    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "output_directory=" + output_directory
    print "input_directory=" + input_directory
    print "corpus_name=" + corpus_name
    # print "dictionary file=" + str(dict_file)
    # parameter set 2
    print "training_iterations=%d" % (training_iterations);
    print "snapshot_interval=" + str(snapshot_interval);
    print "number_of_topics=" + str(number_of_topics)
    # parameter set 3
    print "alpha_alpha=" + str(alpha_alpha)
    print "alpha_beta=" + str(alpha_beta)
    #print "alpha_eta=" + str(alpha_eta)
    print "eta_l2_lambda=" + str(eta_l2_lambda)
    #print "alpha_sigma=" + str(alpha_sigma)
    # parameter set 4
    print "inference_mode=%d" % (inference_mode)
    print "========== ========== ========== ========== =========="
    
    if inference_mode == 0:
        # import hybrid
        # slda_inferencer = hybrid.Hybrid();
        raise NotImplementedError
        return
    elif inference_mode == 1:
        # import monte_carlo
        # slda_inferencer = monte_carlo.MonteCarlo();
        raise NotImplementedError
        return
    elif inference_mode == 2:
        import variational_bayes
        slda_inferencer = variational_bayes.VariationalBayes();
    else:
        sys.stderr.write("error: unrecognized inference mode %d...\n" % (inference_mode));
        return;
    
    slda_inferencer._initialize(train_docs, vocab, labels, number_of_topics, alpha_alpha, alpha_beta, eta_l2_lambda);
    
    for iteration in xrange(training_iterations):
        slda_inferencer.learning();

        if (slda_inferencer._counter % snapshot_interval == 0):
            # slda_inferencer.export_beta(output_directory + 'exp_beta-' + str(slda_inferencer._counter));
            slda_inferencer.export_eta(output_directory + 'exp_eta-' + str(slda_inferencer._counter));
    
    model_snapshot_path = os.path.join(output_directory, 'model-' + str(slda_inferencer._counter));
    cPickle.dump(slda_inferencer, open(model_snapshot_path, 'wb'));
    
if __name__ == '__main__':
    main()
