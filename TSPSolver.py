#!/usr/bin/python3

from which_pyqt import PYQT_VER
from PyQt5.QtCore import QLineF, QPointF

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
import heapq as hq
from collections import OrderedDict
import copy

class bnb_node:
	def __init__( self, nodes_traversed, min_cost_matrix, cur_lower_bound ):
		self.nodes_traversed = nodes_traversed
		self.matrix = min_cost_matrix
		self.cost = cur_lower_bound


class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy( self,time_allowance=60.0 ):
		pass



	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

	def branchAndBound(self, time_allowance=60.0):
		self.counter = itertools.count()  # heap entry count
		results = {}
		self.cities = self._scenario.getCities()
		self.foundTour = False
		self.count = 0
		self.heap = []  # initialize priority queue
		self.bssf = None
		self.num_states = 1  # starts at one ofr the parent node
		self.pruned = 0
		max_heap_size = 0
		start_time = time.time()
		matrix = self.setInitalCostMatrix(self.cities)

		# find a random path to save as initial bssf
		self.bssf = TSPSolution(self.defaultRandomTour().get('soln').route)
		# if self.bssf.cost != float('infinity'):
		# 	self.foundTour = True
		# I am pretty sure we dont want to set this value


		nodes_traversed = [self.cities[0]]
		reduced_matrix, cost = self.get_reduce_cost_matrix(matrix, 0, [])
		parent_node = bnb_node(nodes_traversed, reduced_matrix, cost)
		self.check_children(parent_node)

		while len(self.heap) != 0 and time.time() - start_time < time_allowance:  # taken out for debugging purposes
			# before we pop check the value of the heap and update the max value accordingly
			if len(self.heap) > max_heap_size:
				max_heap_size = len(self.heap)

			heap_tuple = heapq.heappop(self.heap)  # pop the queue
			possible_bssf = heap_tuple[2]
			#self.printFunc(possible_bssf.nodes_traversed, possible_bssf.cost, heap_tuple[0])
			# create a function that prints the letters in the visited nodes list and the total cost

			# check how it compares to BSSF
			if possible_bssf.cost <= self.bssf.cost:
				self.check_children(possible_bssf)  # run check_children again
			else:
				self.pruned = self.pruned + 1


		end_time = time.time()
		results['cost'] = self.bssf.cost if self.foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = self.count
		results['soln'] = self.bssf  # this will be the path, aka the traversedNodes wrapped in a TSPSolution object
		results['max'] = max_heap_size  # max number of stored states at a given time
		results['total'] = self.num_states  # total number of states created
		results['pruned'] = self.pruned  # total number of states pruned
		# this is cities pruned instead of adding it to the queue and pruned after being taken off the queue
		# doesn't include sub-states that weren't created.
		return results

	def check_children(self, parent_node):
		# check if we have found a full path
		cities_remaining = len(self.cities) - len(parent_node.nodes_traversed)
		if cities_remaining == 0:
			self.foundTour = True
			# also check there is a path back to the start and that its cost is accounted for
			parent_node.cost = parent_node.cost + parent_node.matrix[parent_node.nodes_traversed[-1]._index][0]  # the column will always be 0 as long as it starts at A
			if self.bssf.cost > parent_node.cost:
				# print("found a better solution!", self.bssf.cost, parent_node.cost)
				# self.printFunc(parent_node.nodes_traversed,parent_node.cost,"not sure")
				self.bssf.route = parent_node.nodes_traversed
				self.bssf.cost = parent_node.cost
				self.count = self.count + 1
			return
		city_start = parent_node.nodes_traversed[-1]  # get last item
		city_start_index = self.cities.index(city_start)  # this is the row index
		for col_index, city_dest in enumerate(self.cities):
			if city_dest not in parent_node.nodes_traversed:  # skip cities that are already in our nodes_traversed list
				child_matrix = copy.deepcopy(parent_node.matrix)  # copying so we don't override it
				cost = parent_node.cost
				updated_nodes_traversed = parent_node.nodes_traversed + [city_dest]
				# update cost with index (citystart, city end) cost
				cost = cost + child_matrix[city_start_index][col_index]
				updated_cost_matrix = self.set_child_reduced_cost_matrix(child_matrix, city_start_index, col_index)
				updated_reduced_cost_matrix, cost = self.get_reduce_cost_matrix(updated_cost_matrix, cost, updated_nodes_traversed )
				# if the BSSF is worse than the global_bssf immediately prune
				if cost <= self.bssf.cost:
					self.num_states = self.num_states + 1
					childNode = bnb_node(updated_nodes_traversed, updated_reduced_cost_matrix, cost)
					# add the bssf value and divide by the number of remaining cities to explore.
					# this way if you have few trees to explore you hopefully have a smaller number
					index = next(self.counter)  # object index
					priority = cost + cost/ math.factorial(cities_remaining)
					#priority = cost
					#index = cities_remaining
					# print("its smaller ", int(priority), updated_nodes_traversed)
					heapq.heappush(self.heap, (index, priority, childNode))  # adding it to the queue
					# push just pushes, it doesn't insert based of priority,
					# do I need to sort it later?
					#self.printFunc(childNode.nodes_traversed, childNode.cost, priority)

				else:
					self.pruned = self.pruned +1

	def set_child_reduced_cost_matrix(self, matrix, city_start_index, city_end_index):
		# return the reduced cost matrix with all the infinities on the row and columns of the matrix and the opposite index
		matrix[city_start_index] = float('infinity')
		matrix[:, city_end_index] = float('infinity')
		matrix[city_end_index, city_start_index] = float('infinity')
		return matrix

	def get_reduce_cost_matrix(self, matrix, cost, visited_nodes):
		# if a row_index matches the visited_node index, skip (we already know it will be all infinity)
		for row_index, matrix_row in enumerate(matrix):
			city = self.cities[row_index]  # convert row_index to a city node
			if city not in visited_nodes:
				min_num = matrix_row.min()  # the minimum value of this row
				cost = min_num + cost  # update cost
				# if the min_num is infinity and therefore the bound will be infinity
				# we don't want to subtract infinity from infinity because it results in nan
				if min_num != float('infinity'):
					matrix[row_index] = matrix_row - min_num # subtracts the minimum value from the row
				# if the smallest number is infinity, we dont need to do anything to the row

		# if a col_index matches the visted_node index - the first column index (this wont be set yet, because we could still , skip (we already know it will be all infinity)
		for col_index, matrix_col in enumerate(matrix.T):
			city = self.cities[col_index]  # convert col_index to a city node
			if city not in visited_nodes and col_index != 0:
				min_num = matrix_col.min()
				cost = min_num + cost # update bound
				if min_num != float('infinity'):
					matrix[:, col_index] = matrix_col - min_num
		# if the smallest number is infinity, we dont need to do anything to the col
		# update that col with list reduce_list returns

		return matrix, cost

	def setInitalCostMatrix(self, cities):
		matrix = np.zeros((len(cities), len(cities)))
		for row_index, city_start in enumerate(cities):
			for col_index, city_dest in enumerate(cities):
				matrix[row_index][col_index] = city_start.costTo(city_dest)
		return matrix

	def printFunc(self, list, cost, priority):
		# given a list of bnb nodes print out the alphabetical version
		for city in list:
			print(city._name, end = '')
		print(" ",cost, priority)
		pass



	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''

	def fancy( self,time_allowance=60.0 ):
		pass




