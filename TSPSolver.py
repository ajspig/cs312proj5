#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))



import time
import numpy as np
from TSPClasses import *
import heapq
import itertools

class bnb_node:
	def __init__( self, nodes_traversed, cur_lower_bound, min_cost_matrix ):
		self.nodes_traversed   = nodes_traversed
		self.lower_bound  = cur_lower_bound
		self.matrix = min_cost_matrix



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
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = 0  # was none before (not sure why?)
		start_time = time.time()
		matrix = self.setInitalCostMatrix(cities)
		# what is the difference between bssf and bound
		# do we update the bssf with the bound?
		reduced_matrix, bssf = self.get_reduce_cost_matrix(matrix, bssf, [])
		# 2. find a random path to get a max bound (save as global variable)
		# 3. starting with the first city pass it into recursive function
		nodes_traversed = [0] # used to be cities[0] when it was the actual city object, now want just index

		random_results = self.defaultRandomTour()
		max_up_bound = random_results['cost']  # it can't get any worse than this!
		# not sure how to use this yet... should play a part in the pruning probably

		parent_node = bnb_node(nodes_traversed, bssf, reduced_matrix)
		best_solution = self.check_children(parent_node, cities,0)
		# best_solution = best city node which contains the best path traversed to get the lowest bound

		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		pass
		# TODO:
		# this wont quite work yet, because the traversedNodes is a list of cities, not the indexes.
		# do I change it so its to lists? a list of indexes and cities??
		# would there ever be a need to have the actual city in the list or can I always just do the index...
	def check_children(self, parent_node, cities, city_start_index):
		# city_start_index will be the row index
		# look at all the children and generate their bnb node( so find the reduced cost matrix and the updated bssf and add the node traveling to the list of nodes
		# loop through all cities (skipp any that are already in parent_node.nodes_traversed)
		for col_index, city_dest in enumerate(cities):
			if city_dest not in parent_node.nodes_traversed: # want to skip cities that are already in our nodes_traversed list
				bssf = parent_node.lower_bound
				updated_nodes_traversed = parent_node.nodes_traversed + [city_dest]
				# updated_nodes_traversed.append(city_dest)
				updated_cost_matrix = self.set_child_reduced_cost_matrix(parent_node.matrix, city_start_index, col_index)
				updated_reduced_cost_matrix, bssf = self.get_reduce_cost_matrix(updated_cost_matrix, bssf,updated_nodes_traversed )
				# this could be when use the upper bound.
				# because if the BSSF is worse than the upper bound, we already know we don't even want to look at it or add it to the stack.
				# immediately pruned
				childNode = bnb_node(updated_nodes_traversed, updated_reduced_cost_matrix, bssf)
				# where do I store the child? in a list and then after we run through everything just find whichever child has best bssf?


		return parent_node
		# which ever has the best bssf call this function again.
		# for all others add it to the queue
		# also want to do some fancy thing where we figure out what to pop of the BSSF based on the depth that node is in the tree
		# the idea being that the nodes that are further in the tree have a better estimate of the BSSF so should be prioritized

	def set_child_reduced_cost_matrix(self, matrix, city_start_index, city_end_index):
		# return the reduced cost matrix with all the infinities on the row and columns of the matrix and the opposite index
		matrix[city_start_index] = float('infinity')
		matrix[:, city_end_index] = float('infinity')
		matrix[city_end_index, city_start_index] = float('infinity')
		return matrix

# todo: this function is still a little problematic
	# want to make sure that we only update the bound if it is for a row or a column not already in our list of points
	# that way we dont get infinities accidentally added to our bound, when its really because we just marked them all of as infinity
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
			# hopefully the [1:] gets it in a pythonic way
			if col_index not in visited_nodes[1:]:
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




