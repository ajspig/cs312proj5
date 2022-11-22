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
		results = {}
		cities = self._scenario.getCities()
		self.foundTour = False
		count = 0
		self.heap = []  # initialize priority queue
		self.bssf = None
		start_time = time.time()
		matrix = self.setInitalCostMatrix(cities)
		# what is the difference between bssf and bound
		# do we update the bssf with the bound?
		# 2. find a random path to get a max bound (save as global variable)
		self.bssf = TSPSolution(self.defaultRandomTour().get('soln').route)
		if self.bssf.cost != float('infinity'):
			self.foundTour = True

		nodes_traversed = OrderedDict({0: cities[0]})
		reduced_matrix, cost = self.get_reduce_cost_matrix(matrix, 0, {})
		parent_node = bnb_node(nodes_traversed, reduced_matrix, cost)
		self.check_children(parent_node, cities)
		while len(self.heap) != 0:
			self.check_children(heapq.heappop(self.heap), cities)

		# pop the queue and run the check_children again.

		end_time = time.time()
		results['cost'] = self.bssf.cost if self.foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = self.bssf  # this will be the path, aka the traversedNodes
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		pass

	def check_children(self, parent_node, cities):
		# check if we have found a full path
		cities_remaining = len(cities) - len(parent_node.nodes_traversed)
		if cities_remaining == 0:
			self.foundTour = True
			self.bssf = TSPSolution(parent_node.nodes_traversed)
			return

		city_start_index = list(parent_node.nodes_traversed.keys())[-1]  # get last item
		# city_start_index will be the row index
		for col_index, city_dest in enumerate(cities):
			if col_index not in parent_node.nodes_traversed:  # skip cities that are already in our nodes_traversed list
				cost = parent_node.cost
				updated_nodes_traversed = parent_node.nodes_traversed
				updated_nodes_traversed.update({col_index: city_dest})
				updated_cost_matrix = self.set_child_reduced_cost_matrix(parent_node.matrix, city_start_index, col_index)
				updated_reduced_cost_matrix, cost = self.get_reduce_cost_matrix(updated_cost_matrix, cost,updated_nodes_traversed )
				# if the BSSF is worse than the global_bssf immediately prune
				# self.bssf.cost doesn't exist. try again. I think im confusing the difference between the bssf the result from the function, the dictionary the class I made.
				if cost < self.bssf.cost:
					childNode = bnb_node(updated_nodes_traversed, updated_reduced_cost_matrix, cost)
					# maybe add the bssf value + the number of remaining cities to explore
					# this way if you have few trees to explore you hopefully have a smaller number
					priority = cost + cities_remaining
					heapq.heappush(self.heap, (priority, childNode))  # adding it to the queue
				# everytime this isn't true, we increment the total pruned.

	def set_child_reduced_cost_matrix(self, matrix, city_start_index, city_end_index):
		# return the reduced cost matrix with all the infinities on the row and columns of the matrix and the opposite index
		matrix[city_start_index] = float('infinity')
		matrix[:, city_end_index] = float('infinity')
		matrix[city_end_index, city_start_index] = float('infinity')
		return matrix

	def get_reduce_cost_matrix(self, matrix, bound, visited_nodes):
		# if a row_index matches the visited_node index, skip (we already know it will be all infinity)
		for row_index, matrix_row in enumerate(matrix):
			if row_index not in visited_nodes:
				min_num = matrix_row.min() # the minimum value of this row
				bound = min_num + bound # update bound
				# if the min_num is infinity and therefore the bound will be inifinty
				# we dont want to subtract infinity becuase it results in nan
				if min_num != float('infinity'):
					matrix[row_index] = matrix_row - min_num # subtracts the minimum value from the row
				# if the smallest number is infinity, we dont need to do anything to the row

		# if a col_index matches the visted_node index - the first column index (this wont be set yet, because we could still , skip (we already know it will be all infinity)
		for col_index, matrix_col in enumerate(matrix.T):

			if col_index not in visited_nodes.keys() and col_index != 0:
				min_num = matrix_col.min()
				bound = min_num + bound # update bound
				if min_num != float('infinity'):
					matrix[:, col_index] = matrix_col - min_num
		# if the smallest number is infinity, we dont need to do anything to the col
		# update that col with list reduce_list returns

		return matrix, bound

	def setInitalCostMatrix(self, cities):
		# matrix = [[None] * len(cities) for _ in range(len(cities))]
		matrix = np.zeros((len(cities), len(cities)))
		for row_index, city_start in enumerate(cities):
			for col_index, city_dest in enumerate(cities):
				if city_start != city_dest:  # checking if they are the same city
					matrix[row_index][col_index] = city_start.costTo(city_dest)
					# TODO: what does this return if there is no distance between those nodes? want it to return infinity
				else:
					matrix[row_index][col_index] = float("infinity")  # setting the diagonal to infinity
		return matrix



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




