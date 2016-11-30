"""
VariationalBayes for Supervised LDA
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import time
import numpy
import scipy;
import scipy.misc;
import nltk;
import string;
import sys;

from inferencer import compute_dirichlet_expectation
from inferencer import Inferencer;

"""
This is a python implementation of supervised lda, based on variational inference, with hyper parameter updating.

References:
[1] Blei David M., McAuliffe Jon D., Supervised topic models. In Advances in Neural Information Processing Systems, 2008
"""

class VariationalBayes(Inferencer):
    def __init__(self,
                 hyper_parameter_optimize_interval=1,
                 symmetric_alpha_alpha=True,
                 symmetric_alpha_beta=True,
                 #scipy_optimization_method="BFGS",
                 scipy_optimization_method="L-BFGS-B",
                 #scipy_optimization_method = "CG"
                 ):
        Inferencer.__init__(self, hyper_parameter_optimize_interval);

        self._symmetric_alpha_alpha = symmetric_alpha_alpha
        self._symmetric_alpha_beta = symmetric_alpha_beta
        
        self._scipy_optimization_method = scipy_optimization_method

    """
    @param num_topics: the number of topics
    @param data: a defaultdict(dict) data type, first indexed by doc id then indexed by term id
    take note: words are not terms, they are repeatable and thus might be not unique
    """
    def _initialize(self,
                    corpus,
                    vocab,
                    labels,
                    number_of_topics,
                    alpha_alpha,
                    alpha_beta,
                    #alpha_eta=1,
                    eta_l2_lambda=1.0,
                    # alpha_sigma_square=1.0
                    ):
        Inferencer._initialize(self, vocab, labels, number_of_topics, alpha_alpha, alpha_beta);
        
        self._parsed_corpus, self._parsed_labels = self.parse_data(corpus);
        
        # define the total number of document
        self._number_of_documents = len(self._parsed_corpus);
        
        # initialize a D-by-K matrix gamma, valued at N_d/K
        self._gamma = numpy.zeros((self._number_of_documents, self._number_of_topics)) + self._alpha_alpha[numpy.newaxis, :] + 1.0 * self._number_of_types / self._number_of_topics;
        # self._gamma = numpy.random.gamma(100., 1./100, (self._number_of_documents, self._number_of_topics))
        
        # initialize a V-by-K matrix _eta, valued at 1/V, subject to the sum over every row is 1
        self._beta = numpy.random.gamma(100., 1. / 100., (self._number_of_topics, self._number_of_types));
        # self._beta /= numpy.sum(self._beta, 1)[:, numpy.newaxis]
        # self._E_log_eta = compute_dirichlet_expectation(self._beta);
        
        #self._eta = numpy.zeros((self._number_of_labels, self._number_of_topics)) + alpha_eta
        self._eta = numpy.random.random((self._number_of_labels, self._number_of_topics));
        self._eta /= numpy.sqrt(numpy.sum(self._eta**2));

        self._eta_l2_lambda = eta_l2_lambda
        # self._sigma_square = alpha_sigma_square
        
    def e_step(self,
               parsed_corpus_labels=None,
               local_gamma_iteration=10,
               local_phi_iteration=10,
               local_parameter_converge_threshold=1e-6,
               approximate_phi=False):
        
        if parsed_corpus_labels == None:
            word_idss = self._parsed_corpus;
            label_idss = self._parsed_labels
        else:
            word_idss = parsed_corpus_labels;
            label_idss = None;

        number_of_documents = len(word_idss);
        
        document_log_likelihood = 0;
        words_log_likelihood = 0;
        
        # initialize a V-by-K matrix phi sufficient statistics
        phi_sufficient_statistics = numpy.zeros((self._number_of_topics, self._number_of_types));
        E_A_sufficient_statistics = numpy.zeros((number_of_documents, self._number_of_topics))
        E_AA_sufficient_statistics = numpy.zeros((number_of_documents, self._number_of_topics, self._number_of_topics))
        
        # initialize a D-by-K matrix gamma values
        gamma_values = numpy.zeros((number_of_documents, self._number_of_topics)) + self._alpha_alpha[numpy.newaxis, :] + 1.0 * self._number_of_types / self._number_of_topics;
        
        E_log_beta = compute_dirichlet_expectation(self._beta);
        assert E_log_beta.shape == (self._number_of_topics, self._number_of_types);
        if parsed_corpus_labels != None:
            E_log_prob_eta = E_log_beta - scipy.misc.logsumexp(E_log_beta, axis=1)[:, numpy.newaxis]

        for doc_id in xrange(number_of_documents):
            total_word_count = len(word_idss[doc_id]);
            term_ids = word_idss[doc_id];
            if parsed_corpus_labels == None:
                label_ids = label_idss[doc_id];
            
            # initialize gamma for this document
            gamma_values[doc_id, :] = self._alpha_alpha + 1.0 * total_word_count / self._number_of_topics;
            
            log_phi = scipy.special.psi(gamma_values[doc_id, :][numpy.newaxis, :]) + E_log_beta[:, term_ids].T;
            log_phi -= scipy.misc.logsumexp(log_phi, axis=1)[:, numpy.newaxis];
            assert log_phi.shape == (len(term_ids), self._number_of_topics);
            # phi = numpy.exp(log_phi);
            
            assert self._eta.shape == (self._number_of_labels, self._number_of_topics);
            auxilary_variables_per_label = numpy.zeros(len(self._index_to_label));
            # log_auxilary_variables_per_label_token = numpy.zeros((len(self._index_to_label), total_word_count));
            for label_index in self._index_to_label:
                log_sum_phi_exp_eta = scipy.misc.logsumexp(log_phi + self._eta[label_index, :][numpy.newaxis, :] / total_word_count, axis=1);
                assert log_sum_phi_exp_eta.shape == (len(term_ids),)
                # log_auxilary_variables_per_label_token[:, label_index] = log_sum_phi_exp_eta;
                auxilary_variables_per_label[label_index] = numpy.exp(numpy.sum(log_sum_phi_exp_eta));
            
            # update phi and gamma until gamma converges
            for gamma_iteration in xrange(local_gamma_iteration):
                if approximate_phi:
                    '''
                    phi = numpy.exp(log_phi);
                    assert phi.shape == (len(term_ids), self._number_of_topics);
                    
                    phi_sum = numpy.sum(phi, axis=0)[numpy.newaxis, :];
                    phi_sum_j = numpy.tile(phi_sum, (len(term_ids), 1));
                    phi_sum_j -= phi;
                    assert phi_sum_j.shape == (len(term_ids), self._number_of_topics);
                
                    # log_phi = self._E_log_eta[:, term_ids].T + numpy.tile(scipy.special.psi(self._gamma[[doc_id], :]), (len(self._corpus[doc_id]), 1));
                    # log_phi = E_log_beta[:, term_ids].T + numpy.tile(scipy.special.psi(gamma_values[[doc_id], :]), (word_ids[doc_id].shape[0], 1));
                    log_phi = scipy.special.psi(gamma_values[doc_id, :][numpy.newaxis, :]) + E_log_beta[:, term_ids].T;
                    assert log_phi.shape == (len(term_ids), self._number_of_topics);
                    
                    assert self._eta.shape == (1, self._number_of_topics);
                    
                    log_phi += ((label_idss[doc_id] / (total_word_count * self._sigma_square)) * self._eta)
                    assert log_phi.shape == (len(term_ids), self._number_of_topics);
                    
                    log_phi -= (numpy.dot(phi_sum_j, self._eta.T) * self._eta + 0.5 * (self._eta ** 2)) / ((numpy.float(total_word_count) ** 2.) * self._sigma_square)
                    assert log_phi.shape == (len(term_ids), self._number_of_topics);
                    
                    # phi_normalizer = numpy.log(numpy.sum(numpy.exp(log_phi), axis=1)[:, numpy.newaxis]);
                    # assert phi_normalizer.shape == (len(term_ids), 1);
                    # log_phi -= phi_normalizer;
                    log_phi -= scipy.misc.logsumexp(log_phi, axis=1)[:, numpy.newaxis];
                    assert log_phi.shape == (len(term_ids), self._number_of_topics);
                    
                    gamma_update = self._alpha_alpha + numpy.array(numpy.sum(numpy.exp(log_phi), axis=0));
                    mean_change = numpy.mean(abs(gamma_update - gamma_values[doc_id, :]));
                    gamma_values[doc_id, :] = gamma_update;
                    if mean_change <= local_parameter_converge_threshold:
                        break;
                    '''
                    pass
                else:
                    old_gamma_values = gamma_values[doc_id, :].copy();

                    assert log_phi.shape == (len(term_ids), self._number_of_topics);
                    for term_pos in xrange(len(term_ids)):
                        term_id = term_ids[term_pos];
                        
                        h_vector = numpy.zeros(self._number_of_topics);
                        for label_index in self._index_to_label:
                            log_sum_phi_n_exp_eta = scipy.misc.logsumexp(log_phi[term_pos, :] + self._eta[label_index, :] / total_word_count);
                            sum_phi_n_exp_eta = numpy.exp(log_sum_phi_n_exp_eta);
                            # numpy.sum(log_auxilary_variables_per_label_token[:term_pos, label_index]) + numpy.sum(log_auxilary_variables_per_label_token[term_pos:, label_index])
                            auxilary_variables_per_label[label_index] /= sum_phi_n_exp_eta;
                            
                            h_vector += auxilary_variables_per_label[label_index] * numpy.exp(self._eta[label_index, :] / total_word_count)
                        
                        for phi_iteration in xrange(local_phi_iteration):
                            phi_n = numpy.exp(log_phi[term_pos, :]);
                            # log_phi = self._E_log_eta[:, term_ids].T + numpy.tile(scipy.special.psi(self._gamma[[doc_id], :]), (len(self._corpus[doc_id]), 1));
                            # log_phi = E_log_beta[:, term_ids].T + numpy.tile(scipy.special.psi(gamma_values[[doc_id], :]), (word_ids[doc_id].shape[0], 1));
                            log_phi_n = scipy.special.psi(gamma_values[doc_id, :]) + E_log_beta[:, term_id];
                            assert log_phi_n.shape == (self._number_of_topics,);

                            if parsed_corpus_labels == None:
                                log_phi_n += numpy.sum(self._eta[label_ids, :], axis=0) / total_word_count
                                log_phi_n -= len(label_ids) * h_vector / numpy.dot(h_vector, phi_n)
                            else:
                                log_phi_n -= h_vector / numpy.dot(h_vector, phi_n)
                            assert log_phi_n.shape == (self._number_of_topics,);
                            
                            log_phi_n -= scipy.misc.logsumexp(log_phi_n);
                            
                            log_phi[term_pos, :] = log_phi_n;
                        
                        for label_index in self._index_to_label:
                            log_sum_phi_n_exp_eta = scipy.misc.logsumexp(log_phi[term_pos, :] + self._eta[label_index, :] / total_word_count);
                            sum_phi_n_exp_eta = numpy.exp(log_sum_phi_n_exp_eta);
                            # numpy.sum(log_auxilary_variables_per_label_token[:term_pos, label_index]) + numpy.sum(log_auxilary_variables_per_label_token[term_pos:, label_index])
                            auxilary_variables_per_label[label_index] *= sum_phi_n_exp_eta;
                            
                        gamma_values[doc_id, :] = self._alpha_alpha + numpy.array(numpy.sum(numpy.exp(log_phi), axis=0));
                    mean_change = numpy.mean(abs(gamma_values[doc_id, :] - old_gamma_values));
                    if mean_change <= local_parameter_converge_threshold:
                        break;
                
                '''
                # TODO: We could also update the gamma after all phi updates.
                gamma_update = self._alpha_alpha + numpy.array(numpy.sum(numpy.exp(log_phi), axis=0));
                mean_change = numpy.mean(abs(gamma_update - gamma_values[doc_id, :]));
                gamma_values[doc_id, :] = gamma_update;
                if mean_change <= local_parameter_converge_threshold:
                    break;
                '''
        
            phi = numpy.exp(log_phi);
            assert phi.shape == (len(term_ids), self._number_of_topics);
            phi_mean = numpy.mean(phi, axis=0)
            assert phi_mean.shape == (self._number_of_topics,);
            
            # Note: all terms including E_q[p(\theta | \_alpha_alpha)], i.e., terms involving \Psi(\gamma), are cancelled due to \gamma updates in E-step
            
            # compute the _alpha_alpha terms
            document_log_likelihood += scipy.special.gammaln(numpy.sum(self._alpha_alpha)) - numpy.sum(scipy.special.gammaln(self._alpha_alpha))
            # compute the gamma terms
            document_log_likelihood += numpy.sum(scipy.special.gammaln(gamma_values[doc_id, :])) - scipy.special.gammaln(numpy.sum(gamma_values[doc_id, :]));
            # compute the phi terms
            document_log_likelihood -= numpy.sum(phi * log_phi);
            # compute the eta terms
            if parsed_corpus_labels == None:
                document_log_likelihood += numpy.dot(numpy.sum(self._eta[label_ids, :], axis=0), phi_mean);
            document_log_likelihood -= numpy.log(numpy.sum(auxilary_variables_per_label))
            
            # Note: all terms including E_q[p(\_eta | \_beta)], i.e., terms involving \Psi(\_eta), are cancelled due to \_eta updates in M-step
            if parsed_corpus_labels != None:
                # compute the p(w_{dn} | z_{dn}, \_eta) terms, which will be cancelled during M-step during training
                words_log_likelihood += numpy.sum(phi.T * E_log_prob_eta[:, term_ids]);
                
            assert(phi.shape == (len(term_ids), self._number_of_topics));
            for term_pos in xrange(len(term_ids)):
                term_id = term_ids[term_pos];
                phi_sufficient_statistics[:, term_id] += phi[term_pos, :];
            # phi_sufficient_statistics[:, term_ids] += numpy.exp(log_phi + numpy.log(term_counts.transpose())).T;
            
            E_A_sufficient_statistics[doc_id, :] = phi_mean;
            E_AA_sufficient_statistics[doc_id, :, :] = numpy.dot(phi_mean[:, numpy.newaxis], phi_mean[numpy.newaxis, :]);
                
            if (doc_id + 1) % 10 == 0:
                print "successfully processed %d documents..." % (doc_id + 1);
            
        # compute mean absolute error
        # mean_absolute_error = numpy.abs(numpy.dot(E_A_sufficient_statistics, self._eta.T) - label_idss[:, numpy.newaxis]).sum()
        
        if parsed_corpus_labels == None:
            self._gamma = gamma_values;
            return document_log_likelihood, phi_sufficient_statistics, E_A_sufficient_statistics, E_AA_sufficient_statistics
        else:
            return words_log_likelihood, gamma_values, numpy.dot(E_A_sufficient_statistics, self._eta.T)

    # TODO: this is a direct import form Chong Wang's SLDA code, which is significantly different from the paper.
    def optimize_eta(self,
                     eta,
                     arguments,
                    ):
        optimize_result = scipy.optimize.minimize(self.f_eta,
                                                  eta,
                                                  args=arguments,
                                                  method=self._scipy_optimization_method,
                                                  jac=self.f_prime_eta,
                                                  # hess=self.f_hessian_eta,
                                                  # hess=None,
                                                  # hessp=self.f_hessian_direction_eta,
                                                  bounds=None,
                                                  constraints=(),
                                                  tol=None,
                                                  callback=None,
                                                  options={'disp': False}
                                                  )
        
        return optimize_result.x

    def f_eta(self, eta_vector, *args):
        (E_A_sufficient_statistics, E_AA_sufficient_statistics) = args;

        assert eta_vector.shape == (self._number_of_labels * self._number_of_topics,)
        l2_regularizer = numpy.sum(eta_vector ** 2) * self._eta_l2_lambda;
        eta = numpy.reshape(eta_vector, (self._number_of_labels, self._number_of_topics))

        likelihood = 0;
        for doc_id in xrange(len(self._parsed_labels)):
            label_ids = self._parsed_labels[doc_id]

            likelihood += numpy.sum(numpy.dot(eta[label_ids, :], E_A_sufficient_statistics[[doc_id], :].T))

            a1 = numpy.dot(eta, E_A_sufficient_statistics[doc_id, :][:, numpy.newaxis]);
            assert a1.shape == (self._number_of_labels, 1)
            a2 = numpy.zeros((self._number_of_labels, 1));
            for label_id in xrange(self._number_of_labels):
                a2[label_id, 0] = numpy.dot(numpy.dot(eta[[label_id], :], E_AA_sufficient_statistics[doc_id, :, :]), eta[[label_id], :].T) * 0.5 + 1.0
            #a2 = numpy.diag(numpy.dot(numpy.dot(eta, E_AA_sufficient_statistics[doc_id, :, :]), eta.T))[:, numpy.newaxis] * 0.5 + 1.0
            assert a2.shape == (self._number_of_labels, 1)

            likelihood -= scipy.misc.logsumexp(a1 + numpy.log(a2));

        return -likelihood + l2_regularizer;
      
    def f_prime_eta(self, eta_vector, *args):
        (E_A_sufficient_statistics, E_AA_sufficient_statistics) = args;

        assert eta_vector.shape == (self._number_of_labels * self._number_of_topics,)
        l2_regularizer = 2 * self._eta_l2_lambda * eta_vector;
        eta = numpy.reshape(eta_vector, (self._number_of_labels, self._number_of_topics))

        derivative = numpy.zeros((self._number_of_labels, self._number_of_topics))
        assert derivative.shape == (self._number_of_labels, self._number_of_topics)


        for doc_id in xrange(len(self._parsed_labels)):
            label_ids = self._parsed_labels[doc_id]

            for label_id in label_ids:
                derivative[label_id, :] += E_A_sufficient_statistics[doc_id, :];
                # derivative += numpy.dot(eta[label_ids, :], E_A_sufficient_statistics[doc_id, :][numpy.newaxis, :])

            #numpy.tile(numpy.sum(E_A_sufficient_statistics, axis=0), (self._number_of_labels, 1));

            a1 = numpy.dot(eta, E_A_sufficient_statistics[doc_id, :][:, numpy.newaxis]);
            assert a1.shape == (self._number_of_labels, 1), a1.shape
            a2 = numpy.zeros((self._number_of_labels, 1));
            for label_id in xrange(self._number_of_labels):
                a2[label_id, 0] = numpy.dot(numpy.dot(eta[[label_id], :], E_AA_sufficient_statistics[doc_id, :, :]), eta[[label_id], :].T) * 0.5 + 1.0
            #a2 = numpy.diag(numpy.dot(numpy.dot(eta, E_AA_sufficient_statistics[doc_id, :, :]), eta.T))[:, numpy.newaxis] * 0.5 + 1.0
            assert a2.shape == (self._number_of_labels, 1), a2.shape

            t_value = scipy.misc.logsumexp(a1 + numpy.log(a2));

            eta_aux = numpy.dot(eta, E_AA_sufficient_statistics[doc_id, :, :])
            assert eta_aux.shape==(self._number_of_labels, self._number_of_topics);

            #eta_aux = numpy.zeros((self._number_of_labels, self._number_of_topics));
            #for label_id in xrange(self._number_of_labels):
                # eta_aux += numpy.dot(E_AA_sufficient_statistics[doc_id, :, :], eta[label_id, :])
                #eta_aux[label_id, :] = numpy.dot(eta[[label_id], :], E_AA_sufficient_statistics[doc_id, :, :])[0, :]

            derivative_temp = numpy.zeros((self._number_of_labels, self._number_of_topics));
            for label_id in xrange(self._number_of_labels):
                derivative_temp[label_id, :] += -numpy.exp(a1[label_id, 0]) * (a2[label_id, 0] * E_A_sufficient_statistics[doc_id, :] + eta_aux[label_id, :]);
            #derivative_temp = numpy.exp(a1) * (numpy.dot(a2, E_A_sufficient_statistics[doc_id, :][numpy.newaxis, :]) + eta_aux[numpy.newaxis, :]);
            assert derivative_temp.shape == (self._number_of_labels, self._number_of_topics);
            derivative_temp *= numpy.exp(-t_value)

            derivative += derivative_temp

        derivative = numpy.reshape(derivative, (self._number_of_labels * self._number_of_topics,));
        return -derivative + l2_regularizer;
    
    def m_step(self, phi_sufficient_statistics, E_A_sufficient_statistics, E_AA_sufficient_statistics):
        assert phi_sufficient_statistics.shape == (self._number_of_topics, self._number_of_types);
        assert E_A_sufficient_statistics.shape == (self._number_of_documents, self._number_of_topics);
        assert E_AA_sufficient_statistics.shape == (self._number_of_documents, self._number_of_topics, self._number_of_topics);
        
        # Note: all terms including E_q[p(\_eta|\_beta)], i.e., terms involving \Psi(\_eta), are cancelled due to \_eta updates
        
        # compute the _beta terms
        topic_log_likelihood = self._number_of_topics * (scipy.special.gammaln(numpy.sum(self._alpha_beta)) - numpy.sum(scipy.special.gammaln(self._alpha_beta)));
        # compute the _eta terms
        topic_log_likelihood += numpy.sum(numpy.sum(scipy.special.gammaln(self._beta), axis=1) - scipy.special.gammaln(numpy.sum(self._beta, axis=1)));

        self._beta = phi_sufficient_statistics + self._alpha_beta;
        assert(self._beta.shape == (self._number_of_topics, self._number_of_types));
        
        arguments = (E_A_sufficient_statistics, E_AA_sufficient_statistics)
        eta_vector = numpy.reshape(self._eta, (self._number_of_labels * self._number_of_topics,))
        eta_vector = self.optimize_eta(eta_vector, arguments);
        self._eta = numpy.reshape(eta_vector, (self._number_of_labels, self._number_of_topics))
        assert self._eta.shape == (self._number_of_labels, self._number_of_topics);
        
        '''
        assert self._parsed_labels.shape == (self._number_of_documents,);
        self._sigma_square = numpy.dot(self._parsed_labels[numpy.newaxis, :], self._parsed_labels[:, numpy.newaxis])
        y_E_A = numpy.dot(self._parsed_labels[numpy.newaxis, :], E_A_sufficient_statistics)
        assert y_E_A.shape == (1, self._number_of_topics);
        self._sigma_square -= numpy.dot(numpy.dot(y_E_A, numpy.linalg.inv(E_AA_sufficient_statistics)), y_E_A.T);
        self._sigma_square /= self._number_of_documents;
        '''
        
        # compute the sufficient statistics for _alpha_alpha and update
        alpha_sufficient_statistics = scipy.special.psi(self._gamma) - scipy.special.psi(numpy.sum(self._gamma, axis=1)[:, numpy.newaxis]);
        alpha_sufficient_statistics = numpy.sum(alpha_sufficient_statistics, axis=0);  # [numpy.newaxis, :];
        
        return topic_log_likelihood, alpha_sufficient_statistics

    """
    """
    def learning(self):
        self._counter += 1;
        
        clock_e_step = time.time();
        document_log_likelihood, phi_sufficient_statistics, E_A_sufficient_statistics, E_AA_sufficient_statistics = self.e_step();
        clock_e_step = time.time() - clock_e_step;
        
        clock_m_step = time.time();
        topic_log_likelihood, alpha_sufficient_statistics = self.m_step(phi_sufficient_statistics, E_A_sufficient_statistics, E_AA_sufficient_statistics);
        if self._hyper_parameter_optimize_interval > 0 and self._counter % self._hyper_parameter_optimize_interval == 0:
            self.optimize_hyperparameters(alpha_sufficient_statistics);
        clock_m_step = time.time() - clock_m_step;
        
        joint_log_likelihood = document_log_likelihood + topic_log_likelihood;
        
        print "e_step and m_step of iteration %d finished in %d and %d seconds respectively with log likelihood %g" % (self._counter, clock_e_step, clock_m_step, joint_log_likelihood)
        # print self._eta
        
        return joint_log_likelihood
    
    def inference(self, corpus):
        parsed_corpus, parsed_labels = self.parse_data(corpus);
        
        clock_e_step = time.time();
        words_log_likelihood, corpus_gamma_values, predicted_responses = self.e_step(parsed_corpus);
        assert predicted_responses.shape==(len(parsed_corpus), self._number_of_labels);
        clock_e_step = time.time() - clock_e_step;

        if len(parsed_labels) == 0:
            return words_log_likelihood, corpus_gamma_values, predicted_responses
        elif len(parsed_labels) == len(parsed_corpus):
            return words_log_likelihood, corpus_gamma_values, predicted_responses, parsed_labels
    
    """
    @param alpha_vector: a dict data type represents dirichlet prior, indexed by topic_id
    @param alpha_sufficient_statistics: a dict data type represents _alpha_alpha sufficient statistics for _alpha_alpha updating, indexed by topic_id
    """
    def optimize_hyperparameters(self, alpha_sufficient_statistics, hyper_parameter_iteration=50, hyper_parameter_decay_factor=0.9, hyper_parameter_maximum_decay=10, hyper_parameter_converge_threshold=1e-6):
        # assert(alpha_sufficient_statistics.shape == (1, self._number_of_topics));
        assert (alpha_sufficient_statistics.shape == (self._number_of_topics,));
        alpha_update = self._alpha_alpha;
        
        decay = 0;
        for alpha_iteration in xrange(hyper_parameter_iteration):
            alpha_sum = numpy.sum(self._alpha_alpha);
            alpha_gradient = self._number_of_documents * (scipy.special.psi(alpha_sum) - scipy.special.psi(self._alpha_alpha)) + alpha_sufficient_statistics;
            alpha_hessian = -self._number_of_documents * scipy.special.polygamma(1, self._alpha_alpha);

            if numpy.any(numpy.isinf(alpha_gradient)) or numpy.any(numpy.isnan(alpha_gradient)):
                print "illegal alpha gradient vector", alpha_gradient

            sum_g_h = numpy.sum(alpha_gradient / alpha_hessian);
            sum_1_h = 1.0 / alpha_hessian;

            z = self._number_of_documents * scipy.special.polygamma(1, alpha_sum);
            c = sum_g_h / (1.0 / z + sum_1_h);

            # update the _alpha_alpha vector
            while True:
                singular_hessian = False

                step_size = numpy.power(hyper_parameter_decay_factor, decay) * (alpha_gradient - c) / alpha_hessian;
                # print "step size is", step_size
                assert(self._alpha_alpha.shape == step_size.shape);
                
                if numpy.any(self._alpha_alpha <= step_size):
                    singular_hessian = True
                else:
                    alpha_update = self._alpha_alpha - step_size;
                
                if singular_hessian:
                    decay += 1;
                    if decay > hyper_parameter_maximum_decay:
                        break;
                else:
                    break;
                
            # compute the _alpha_alpha sum
            # check the _alpha_alpha converge criteria
            mean_change = numpy.mean(abs(alpha_update - self._alpha_alpha));
            self._alpha_alpha = alpha_update;
            if mean_change <= hyper_parameter_converge_threshold:
                break;

        return

    def export_beta(self, exp_beta_path, top_display=-1):
        output = open(exp_beta_path, 'w');
        E_log_eta = compute_dirichlet_expectation(self._beta);
        for topic_index in xrange(self._number_of_topics):
            output.write("==========\t%d\t==========\n" % (topic_index));
            
            beta_probability = numpy.exp(E_log_eta[topic_index, :] - scipy.misc.logsumexp(E_log_eta[topic_index, :]));

            i = 0;
            for type_index in reversed(numpy.argsort(beta_probability)):
                i += 1;
                output.write("%s\t%g\n" % (self._index_to_type[type_index], beta_probability[type_index]));
                if top_display > 0 and i >= top_display:
                    break;
                
        output.close();
        
    def export_eta(self, exp_eta_path, top_display=-1):
        output = open(exp_eta_path, 'w');
        E_log_eta = compute_dirichlet_expectation(self._beta);
        
        for topic_index in numpy.argsort(self._eta)[0, :]:
            output.write("==========\t%d\t%f\t==========\n" % (topic_index, self._eta[0, topic_index]));
            
            beta_probability = numpy.exp(E_log_eta[topic_index, :] - scipy.misc.logsumexp(E_log_eta[topic_index, :]));

            i = 0;
            for type_index in reversed(numpy.argsort(beta_probability)):
                i += 1;
                output.write("%s\t%g\n" % (self._index_to_type[type_index], beta_probability[type_index]));
                if top_display > 0 and i >= top_display:
                    break;
                
        output.close();

if __name__ == "__main__":
    print "not implemented..."
    
