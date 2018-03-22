from __future__ import division
import numpy as np
import scipy.stats
from scipy.special import erfc, gammaln
import pickle
from copy import deepcopy

class Document(object):
	def __init__(self, index, timestamp, word_distribution, word_count):
		super(Document, self).__init__()
		self.index = index
		self.timestamp = timestamp
		self.word_distribution = word_distribution
		self.word_count = word_count
		
class Cluster(object):
	def __init__(self, index):# alpha, word_distribution, documents, word_count):
		super(Cluster, self).__init__()
		self.index = index
		self.alpha = None
		self.word_distribution = None
		self.word_count = 0

	def add_document(self, doc):
		if self.word_distribution is None:
			self.word_distribution = np.copy(doc.word_distribution)
		else:
			self.word_distribution += doc.word_distribution
		self.word_count += doc.word_count

	def __repr__(self):
		return 'cluster index:' + str(self.index) + '\n' + 'document index:' + str(self.documents) + '\n' +'word_count: ' + str(self.word_count) \
		+ '\nalpha:' + str(self.alpha)

class Particle(object):
	"""docstring for Particle"""
	def __init__(self, weight):
		super(Particle, self).__init__()
		self.weight = weight
		self.log_update_prob = 0
		self.clusters = {} # can be store in the process for efficient memory implementation, key = cluster_index, value = particle object
		self.docs2cluster_ID = [] # the element is the cluster index of a sequence of document ordered by the index of document
		self.active_clusters = {} # dict key = cluster_index, value = list of timestamps in specific cluster (queue)
		self.cluster_num_by_now = 0

	def __repr__(self):
		return 'particle document list to cluster IDs: ' + str(self.docs2cluster_ID) + '\n' + 'weight: ' + str(self.weight)
		

def dirichlet(prior):
	''' Draw 1-D samples from a dirichlet distribution to multinomial distritbution. Return a multinomial probability distribution.
		@param:
			1.prior: Parameter of the distribution (k dimension for sample of dimension k).
		@rtype: 1-D numpy array
	'''
	return np.random.dirichlet(prior).squeeze()

def multinomial(exp_num, probabilities):
	''' Draw samples from a multinomial distribution.
		@param:
			1. exp_num: Number of experiments.
			2. probabilities: multinomial probability distribution (sequence of floats).
		@rtype: 1-D numpy array
	'''
	return np.random.multinomial(exp_num, probabilities).squeeze()

def EfficientImplementation(tn, reference_time, bandwidth, epsilon = 1e-5):
	''' return the time we need to compute to update the triggering kernel
		@param:
			1.tn: float, current document time
			2.reference_time: list, reference_time for triggering_kernel
			3.bandwidth: int, bandwidth for triggering_kernel
			4.epsilon: float, error tolerance
		@rtype: float
	'''
	max_ref_time = max(reference_time)
	max_bandwidth = max(bandwidth)
	tu = tn - ( max_ref_time + np.sqrt( -2 * max_bandwidth * np.log(0.5 * epsilon * np.sqrt(2 * np.pi * max_bandwidth**2)) ))
	return tu

def log_Dirichlet_CDF(outcomes, prior):
	''' the function only applies to the symmetry case when all prior equals to 1.
		@param:
			1.outcomes: output variables vector
			2.prior: must be list of 1's in our case, avoiding the integrals.
		@rtype: 
	'''
	return np.sum(np.log(outcomes)) + scipy.stats.dirichlet.logpdf(outcomes, prior)

def RBF_kernel(reference_time, time_interval, bandwidth):
	''' RBF kernel for Hawkes process.
		@param:
			1.reference_time: np.array, entries larger than 0.
			2.time_interval: float/np.array, entry must be the same.
			3. bandwidth: np.array, entries larger than 0.
		@rtype: np.array
	'''
	numerator = - (time_interval - reference_time) ** 2 / (2 * bandwidth ** 2) 
	denominator = (2 * np.pi * bandwidth ** 2 ) ** 0.5
	return np.exp(numerator) / denominator

def triggering_kernel(alpha, reference_time, time_intervals, bandwidth):
	''' triggering kernel for Hawkes porcess.
		@param:
			1. alpha: np.array, entres larger than 0
			2. reference_time: np.array, entries larger than 0.
			3. time_intervals: float/np.array, entry must be the same.
			4. bandwidth: np.array, entries larger than 0.
		@rtype: np.array
	'''
	#if len(alpha) != len(reference_time):
		#raise Exception("length of alpha and length of reference time must equal")
	time_intervals = time_intervals.reshape(-1, 1)
	#print((alpha * RBF_kernel(reference_time, time_intervals, bandwidth)).shape)
	if len(alpha.shape) == 3:
		return np.sum(np.sum(alpha * RBF_kernel(reference_time, time_intervals, bandwidth), axis = 1), axis = 1)
	else:
		return np.sum(np.sum(alpha * RBF_kernel(reference_time, time_intervals, bandwidth), axis = 0), axis = 0)

def g_theta(timeseq, reference_time, bandwidth, max_time):
	''' g_theta for DHP
		@param:
			2. timeseq: 1-D np array time sequence before current time
			3. base_intensity: float
			4. reference_time: 1-D np.array
			5. bandwidth: 1-D np.array
		@rtype: np.array, shape(3,)
	'''
	timeseq = timeseq.reshape(-1, 1)
	results = 0.5 * ( erfc(- reference_time / (2 * bandwidth ** 2) ** 0.5) - erfc( (max_time - timeseq - reference_time) / (2 * bandwidth ** 2) **0.5) )
	return np.sum(results, axis = 0)

def update_triggering_kernel(timeseq, alphas, reference_time, bandwidth, base_intensity, max_time, log_priors):
	''' procedure of triggering kernel for SMC
		@param:
			1. timeseq: list, time sequence including current time
			2. alphas: 2-D np.array with shape (sample number, length of alpha)
			3. reference_time: np.array
			4. bandwidth: np.array
			5. log_priors: 1-D np.array with shape (sample number,), p(alpha, alpha_0)
			6. base_intensity: float
			7. max_time: float
		@rtype: 1-D numpy array with shape (length of alpha0,)
	'''
	#print(alphas.shape)
	logLikelihood = log_likelihood(timeseq, alphas, reference_time, bandwidth, base_intensity, max_time)
	log_update_weight = log_priors + logLikelihood
	log_update_weight = log_update_weight - np.max(log_update_weight) # avoid overflow
	update_weight = np.exp(log_update_weight); update_weight = update_weight / np.sum(update_weight)
	update_weight = update_weight.reshape(-1,1)
	alpha = np.sum(update_weight * alphas, axis = 0)
	return alpha

def log_likelihood(timeseq, alphas, reference_time, bandwidth, base_intensity, max_time):
	''' compute log_likelihood for a time sequence for a cluster for SMC
		@param:
			1. timeseq: list, time sequence including current time
			2. alphas: 2-D np.array with shape (sample number, length of alpha)
			3. reference_time: np.array
			4. bandwidth: np.array
			5. log_priors: 1-D np.array, p(alpha, alpha_0)
			6. base_intensity: float
			7. max_time: float
		@rtype: 1-D numpy array with shape (sample number,)
	'''
	Lambda_0 = base_intensity * max_time
	alphas_times_gtheta = np.sum(alphas * g_theta(timeseq, reference_time, bandwidth, max_time), axis = 1) # shape = (sample number,)
	if len(timeseq) == 1:
		raise Exception('The length of time sequence must be larger than 1.')
	time_intervals =  timeseq[-1] - timeseq[:-1]
	alphas = alphas.reshape(-1, 1, alphas.shape[-1])
	triggers = np.log(triggering_kernel(alphas, reference_time, time_intervals, bandwidth))
	return -Lambda_0-alphas_times_gtheta+triggers

def log_dirichlet_multinomial_distribution(cls_word_distribution, doc_word_distribution, cls_word_count, doc_word_count, vocabulary_size, priors):
	''' compute the log dirichlet multinomial distribution
		@param:
			1. cls_word_distribution: 1-D numpy array, including document word_distribution
			2. doc_word_distribution: 1-D numpy array
			3. cls_word_count: int, including document word_distribution
			4. doc_word_count: int
			5. vocabulary_size: int
			6. priors: 1-d np.array
		@rtype: float
	'''
	priors_sum = np.sum(priors)
	log_prob = 0
	log_prob += gammaln(cls_word_count - doc_word_count + priors_sum)
	log_prob -= gammaln(cls_word_count + priors_sum)
	log_prob += np.sum(gammaln(cls_word_distribution + priors))
	log_prob -= np.sum(gammaln(cls_word_distribution - doc_word_distribution + priors))
	return log_prob

def test_dirichlet():
	alpha = dirichlet(np.array([1]* 10))
	sample_alpha_list = [dirichlet([1]* 10) for _ in range(3000)]
	print('len(sample_alpha_list)',len(sample_alpha_list))
	print(np.sum(sample_alpha_list[0]))

def test_multinomial():
	probabilities = dirichlet(np.array([1]* 10))
	result = multinomial(5, probabilities)
	print(result)

def test_EfficientImplementation():
	tu = EfficientImplementation(100, [3,7,11], [2,5,10])
	print(tu)

def test_log_Dirichlet_CDF():
	prior = np.array([1]*10)
	outcomes = dirichlet(prior)
	print(outcomes)
	print(log_Dirichlet_CDF(outcomes, prior))

def test_RBF_kernel():
	refernce_time = np.array([3, 7, 11])
	bandwidth = np.array([5, 5, 5])
	time_intervals = 3
	print(RBF_kernel(refernce_time, time_intervals, bandwidth))
	print(RBF_kernel(11,3,5))

def test_triggering_kernel():
	reference_time = np.array([3, 7, 11])
	bandwidth = np.array([5, 5, 5])
	time_intervals = np.array([1, 3])
	time_intervals = time_intervals.reshape(-1, 1)
	print(time_intervals.shape)
	alpha = dirichlet([1] * 3)
	print(alpha)
	print(RBF_kernel(reference_time, time_intervals, bandwidth))
	print(triggering_kernel(alpha, reference_time, time_intervals, bandwidth))
	time_intervals = np.array([1, 3, 50])
	print(triggering_kernel(alpha, reference_time, time_intervals, bandwidth))


def test_g_theta():
	timeseq = np.arange(0.2, 1000000, 0.6)
	bandwidth = np.array([5, 5, 5])
	reference_time = np.array([3, 7, 11])
	current_time = timeseq[-1]
	T = current_time + 1
	output = g_theta(timeseq, reference_time, bandwidth, T)

def test_log_likelihood():
	timeseq = np.arange(0.2, 1000, 0.6)
	alpha0 = np.array([1, 1, 1])
	bandwidth = np.array([5, 5, 5])
	reference_time = np.array([3, 7, 11])
	sample_num = 1000
	current_time = timeseq[-1]
	T = current_time + 1
	base_intensity = 1

	alphas = []
	log_priors = []
	for _ in range(sample_num):
		alpha = dirichlet(alpha0)
		log_prior = log_Dirichlet_CDF(alpha, alpha0)
		alphas.append(alpha)
		log_priors.append(log_prior)

	alphas = np.array(alphas)
	log_priors = np.array(log_priors)

	logLikelihood = log_likelihood(timeseq, alphas, reference_time, bandwidth, base_intensity, T)
	print(logLikelihood)

def test_update_triggering_kernel():
	# generate parameters
	timeseq = np.arange(0.2, 1000, 0.1)
	alpha0 = np.array([1, 1, 1])
	bandwidth = np.array([5, 5, 5])
	reference_time = np.array([3, 7, 11])
	sample_num = 3000
	base_intensity = 1
	current_time = timeseq[-1]
	T = current_time + 1

	alphas = []
	log_priors = []
	for _ in range(sample_num):
		alpha = dirichlet(alpha0)
		log_prior = log_Dirichlet_CDF(alpha, alpha0)
		alphas.append(alpha)
		log_priors.append(log_prior)

	alphas = np.array(alphas)
	log_priors = np.array(log_priors)

	alpha = update_triggering_kernel(timeseq, alphas, reference_time, bandwidth, base_intensity, T, log_priors)
	print(alpha)

def test_log_dirichlet_multinomial_distribution():
	with open('./data/meme/meme_docs.pkl', 'rb') as r:
		documents = pickle.load(r)

	cls_word_distribution = documents[0].word_distribution +  documents[1].word_distribution
	doc_word_distribution  = documents[1].word_distribution
	cls_word_count = documents[0].word_count + documents[1].word_count
	doc_word_count = documents[1].word_count
	vocabulary_size = len(documents[0].word_distribution)
	priors = np.array([1] * vocabulary_size)
	print('cls_word_count', cls_word_count)
	print('doc_word_count', doc_word_count)
	logprob = log_dirichlet_multinomial_distribution(cls_word_distribution, doc_word_distribution, cls_word_count, doc_word_count, vocabulary_size, priors)
	print(logprob)
	
def main():
	#test_update_triggering_kernel()
	#test_log_likelihood()

	'''
	tu = 75
	active_clusters = {1:[50, 60, 76, 100], 2:[10,20,30,40], 3:[100,2000]}
	print(active_clusters)

	for cluster_index in active_clusters.keys():
		timeseq = active_clusters[cluster_index]
		active_timeseq = [t for t in timeseq if t > tu]
		if not active_timeseq:
			del active_clusters[cluster_index]
		else:
			active_clusters[cluster_index] = active_timeseq

	print(active_clusters)
	'''

	test_log_dirichlet_multinomial_distribution()




if __name__ == '__main__':
	main()