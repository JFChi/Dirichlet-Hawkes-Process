from __future__ import print_function
from __future__ import division
import pickle
import numpy as np
from utils import *
import concurrent.futures
from functools import partial
import copy_reg
import types
from copy import deepcopy
import json

def _pickle_method(m):
	if m.im_self is None:
		return getattr, (m.im_class, m.im_func.func_name)
	else:
		return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

class Dirichlet_Hawkes_Process(object):
	"""docstring for Dirichlet Hawkes Prcess"""
	def __init__(self, particle_num, base_intensity, theta0, alpha0, reference_time, vocabulary_size, bandwidth, sample_num):
		super(Dirichlet_Hawkes_Process, self).__init__()
		self.particle_num = particle_num
		self.base_intensity = base_intensity
		self.theta0 = theta0
		self.alpha0 = alpha0
		self.reference_time = reference_time
		self.vocabulary_size = vocabulary_size
		self.bandwidth = bandwidth
		self.sample_num = sample_num
		# initilize particles
		self.particles = []
		for i in range(particle_num):
			self.particles.append(Particle(weight = 1.0 / self.particle_num))
		alphas = []; log_priors = []
		for _ in range(sample_num):
			alpha = dirichlet(alpha0); log_prior = log_Dirichlet_CDF(alpha, alpha0)
			alphas.append(alpha); log_priors.append(log_prior)
		self.alphas = np.array(alphas)
		self.log_priors = np.array(log_priors)
		self.active_interval = None # [tu, tn]

	def sequential_monte_carlo(self, doc, threshold):
		print('\n\nhandling document %d' %doc.index)
		if isinstance(doc, Document): # deal with the case of exact timing
			# get active interval (globally)
			tu = EfficientImplementation(doc.timestamp, self.reference_time, self.bandwidth)
			self.active_interval = [tu, doc.timestamp]
			print('active_interval',self.active_interval)
			

			#sequential
			particles = []
			for particle in self.particles:
				particles.append(self.particle_sampler(particle, doc))

			self.particles = particles
			

			'''
			partial_particle_sampler = partial(self.particle_sampler, doc = doc)
			with concurrent.futures.ProcessPoolExecutor(max_workers = self.particle_num) as executor:
				self.particles = list(executor.map(partial_particle_sampler, self.particles))
			'''

			'''
			partial_particle_sampler = partial(self.particle_sampler, doc = doc)

			with concurrent.futures.ProcessPoolExecutor(max_workers = self.particle_num) as executor:
				try:
					self.particles = list(executor.map(partial_particle_sampler, self.particles))
				except KeyError:
					print('Encounter exception KeyError',len(self.particles))
			'''

			'''
			particle_index = 0
			partial_particle_sampler = partial(self.particle_sampler, doc = doc)
			executor = concurrent.futures.ProcessPoolExecutor(max_workers = self.particle_num)
			wait_for = [executor.submit(partial_particle_sampler, particle) for particle in self.particles]
			concurrent.futures.wait(wait_for)
			particles = []
			for f in concurrent.futures.as_completed(wait_for):
				particle = f.result()
				particles.append(particle)
			self.particles = particles
			'''
				
				#particle_generator = executor.map(partial_particle_sampler, self.particles)
			# begin particles normalization and resampling
			#for i, particle in enumerate(particle_generator):
				#self.particles[i] = particle
			self.particles = self.particles_normal_resampling(self.particles, threshold)

		else: # deal with the case of exact timing
			print('deal with the case of exact timing')

	def particle_sampler(self, particle, doc):
		# sampling cluster label
		particle, selected_cluster_index = self.sampling_cluster_label(particle, doc) #print(selected_cluster_index)
		# update the triggering kernel
		particle.clusters[selected_cluster_index].alpha = self.parameter_estimation(particle, selected_cluster_index)#;print('selected_cluster_index',selected_cluster_index,'alpha', particle.clusters[selected_cluster_index].alpha)
		# calculate the weight update probability
		particle.log_update_prob = self.calculate_particle_log_update_prob(particle, selected_cluster_index, doc)#; print('particle.log_update_prob',particle.log_update_prob)
		return particle

	def sampling_cluster_label(self, particle, doc):
		if len(particle.clusters) == 0: # the case of the first document comes
			# sample cluster label
			particle.cluster_num_by_now += 1
			selected_cluster_index = particle.cluster_num_by_now
			selected_cluster = Cluster(index = selected_cluster_index)
			selected_cluster.add_document(doc)
			particle.clusters[selected_cluster_index] = selected_cluster #.append(selected_cluster)
			particle.docs2cluster_ID.append(selected_cluster_index)
			# update active cluster
			particle.active_clusters = self.update_active_clusters(particle)

		else: # the case of the following document to come
			active_cluster_indexes = [0] # zero for new cluster
			active_cluster_rates = [self.base_intensity]
			# first update the active cluster
			particle.active_clusters = self.update_active_clusters(particle)
			# then calculate rates for each cluster in active interval
			for active_cluster_index, timeseq in particle.active_clusters.iteritems():
				active_cluster_indexes.append(active_cluster_index)
				time_intervals = doc.timestamp - np.array(timeseq)
				alpha = particle.clusters[active_cluster_index].alpha
				rate = triggering_kernel(alpha, self.reference_time, time_intervals, self.bandwidth)
				active_cluster_rates.append(rate)
				#print(type(active_cluster_index),active_cluster_index, time_intervals, type(time_intervals), rate)
			print('active_cluster_indexes', active_cluster_indexes)
			print('active_cluster_rates', active_cluster_rates)
			cluster_selection_probs = np.array(active_cluster_rates)/np.sum(active_cluster_rates)
			#print('cluster_selection_probs', cluster_selection_probs)
			np.random.seed()
			selected_cluster_array = multinomial(exp_num = 1, probabilities = cluster_selection_probs)
			selected_cluster_index = np.array(active_cluster_indexes)[np.nonzero(selected_cluster_array)][0]
			#print('np.nonzero(selected_cluster_array)',np.nonzero(selected_cluster_array))
			#print('selected_cluster_array',selected_cluster_array)
			print('selected_cluster_index', selected_cluster_index)
			#print('type(selected_cluster_index)', type(selected_cluster_index))
			if selected_cluster_index == 0: # the case of new cluster
				particle.cluster_num_by_now += 1
				selected_cluster_index = particle.cluster_num_by_now
				selected_cluster = Cluster(index = selected_cluster_index)
				selected_cluster.add_document(doc)
				particle.clusters[selected_cluster_index] = selected_cluster
				particle.docs2cluster_ID.append(selected_cluster_index)
				particle.active_clusters[selected_cluster_index] = [self.active_interval[1]] # create a new list containing the current time
				#print('active_clusters', particle.active_clusters); print('cluster_num_by_now', particle.cluster_num_by_now) # FOR DEBUG
			else: # the case of the previous used cluster, update active cluster and add document to cluster
				selected_cluster = particle.clusters[selected_cluster_index]
				selected_cluster.add_document(doc)
				particle.docs2cluster_ID.append(selected_cluster_index)
				particle.active_clusters[selected_cluster_index].append(self.active_interval[1])
				#print('active_clusters', particle.active_clusters); print('cluster_num_by_now', particle.cluster_num_by_now) # FOR DEBUG
		return particle, selected_cluster_index

	def parameter_estimation(self, particle, selected_cluster_index):
		#print('updating triggering kernel ...')
		#print(particle.active_clusters[selected_cluster_index])
		timeseq = np.array( particle.active_clusters[selected_cluster_index] )
		if len(timeseq) == 1: # the case of first document in a brand new cluster
			np.random.seed()
			alpha = dirichlet(self.alpha0)
			return alpha
		T = self.active_interval[1] + 1 #;print('updating triggering kernel ..., len(timeseq)', len(timeseq))
		alpha = update_triggering_kernel(timeseq, self.alphas, self.reference_time, self.bandwidth, self.base_intensity, T, self.log_priors)
		return alpha

	def update_active_clusters(self, particle):
		if not particle.active_clusters: # the case of the first document comes
			particle.active_clusters[1] = [self.active_interval[1]]
		else: # update the active clusters
			tu = self.active_interval[0]
			for cluster_index in particle.active_clusters.keys():
				timeseq = particle.active_clusters[cluster_index]
				active_timeseq = [t for t in timeseq if t > tu]
				if not active_timeseq:
					del particle.active_clusters[cluster_index]
				else:
					particle.active_clusters[cluster_index] = active_timeseq
		return particle.active_clusters
	
	def calculate_particle_log_update_prob(self, particle, selected_cluster_index, doc):
		print('calculate_particle_log_update_prob') #print('id(particle.clusters[selected_cluster_index])', id(particle.clusters[selected_cluster_index]));#print('id(particle.clusters[selected_cluster_index]).word_distribution', id(particle.clusters[selected_cluster_index].word_distribution))
		cls_word_distribution = particle.clusters[selected_cluster_index].word_distribution
		cls_word_count = particle.clusters[selected_cluster_index].word_count
		doc_word_distribution = doc.word_distribution
		doc_word_count = doc.word_count
		assert doc_word_count == np.sum(doc.word_distribution)
		assert cls_word_count == np.sum(particle.clusters[selected_cluster_index].word_distribution)
		log_update_prob = log_dirichlet_multinomial_distribution(cls_word_distribution, doc_word_distribution,\
		 cls_word_count, doc_word_count, self.vocabulary_size, self.theta0)#print('particle.log_update_probs',particle.log_update_probs)
		print('log_update_prob', log_update_prob)
		return log_update_prob

	def particles_normal_resampling(self, particles, threshold):
		print('\nparticles_normal_resampling')
		weights = []; log_update_probs = []
		for particle in particles:
			weights.append(particle.weight)
			log_update_probs.append(particle.log_update_prob)
		weights = np.array(weights); log_update_probs = np.array(log_update_probs); print('weights before update:', weights); print('log_update_probs', log_update_probs)
		log_update_probs = log_update_probs - np.max(log_update_probs) # prevent overflow
		update_probs = np.exp(log_update_probs); #print('update_probs',update_probs)
		weights = weights * update_probs #update 
		weights = weights / np.sum(weights) # normalization
		resample_num = len(np.where(weights + 1e-5 < threshold)[0])
		print('weights:', weights) #; print('log_update_probs',log_update_probs);
		print('resample_num:', resample_num)
		if resample_num == 0: #no need to resample particle, but still need to assign the updated weights to paricle weight
			for i, particle in enumerate(particles):
				particle.weight = weights[i]
			return particles
		else:
			remaining_particles = [particle for i, particle in enumerate(particles) if weights[i] + 1e-5 > threshold ]
			resample_probs = weights[np.where(weights > threshold + 1e-5)]; resample_probs = resample_probs/np.sum(resample_probs)
			remaining_particle_weights = weights[np.where(weights > threshold + 1e-5)]
			for i,_ in enumerate(remaining_particles):
				remaining_particles[i].weight = remaining_particle_weights[i]
			np.random.seed()
			resample_distribution = multinomial(exp_num = resample_num, probabilities = resample_probs)#print('len(remaining_particles)', len(remaining_particles)) #print('resample_probs', resample_probs)
			if not resample_distribution.shape: # the case of only one particle left
				for _ in range(resample_num):
					new_particle = deepcopy(remaining_particles[0])
					remaining_particles.append(new_particle)
			else: # the case of more than one particle left
				for i, resample_times in enumerate(resample_distribution):
					for _ in range(resample_times):
						new_particle = deepcopy(remaining_particles[i])
						remaining_particles.append(new_particle)
			# normalize the particle weight again
			update_weights = np.array([particle.weight for particle in remaining_particles]); update_weights = update_weights / np.sum(update_weights)
			for i, particle in enumerate(remaining_particles):
				particle.weight = update_weights[i]
			print('update_weights aftering resampling', update_weights)
			assert np.abs(np.sum(update_weights) - 1) < 1e-5
			assert len(remaining_particles) == self.particle_num
			return remaining_particles

def parse_newsitem_2_doc(news_item, vocabulary_size):
	''' convert (id, timestamp, word_distribution, word_count) to the form of document
	'''
	#print(news_item)
	index = news_item[0]
	timestamp = news_item[1] / 3600.0 # unix time in hour
	word_id = news_item [2][0]
	count = news_item[2][1]
	word_distribution = np.zeros(vocabulary_size)
	word_distribution[word_id] = count
	word_count = news_item[3]
	doc = Document(index, timestamp, word_distribution, word_count)
	# assert doc.word_count == np.sum(doc.word_distribution)
	return doc

def main():
	with open('../all_the_news_2017.json') as f:
		news_items = json.load(f)
	print('finish extracting news from json...')
	
	# parameter initialization
	vocabulary_size = 56720
	particle_num = 16
	base_intensity = 0.1
	theta0 = np.array([0.01] * vocabulary_size)
	alpha0 = np.array([0.1] * 6)
	reference_time = np.array([0.5, 1, 8, 12, 24, 48])
	bandwidth = np.array([1, 1, 8, 12, 12, 24])
	sample_num = 2000
	threshold = 1.0 / particle_num

	DHP = Dirichlet_Hawkes_Process(particle_num = particle_num, base_intensity = base_intensity, theta0 = theta0, alpha0 = alpha0, \
		reference_time = reference_time, vocabulary_size = vocabulary_size, bandwidth = bandwidth, sample_num = sample_num)

	# begin sampling
	# for simple experiment
	news_items = news_items[:20000]
	for news_item in news_items:
		doc = parse_newsitem_2_doc(news_item = news_item, vocabulary_size = vocabulary_size)
		DHP.sequential_monte_carlo(doc, threshold)


	with open('./result/particles.pkl', 'wb') as w:
		pickle.dump(DHP.particles, w)

if __name__ == '__main__':
	main()