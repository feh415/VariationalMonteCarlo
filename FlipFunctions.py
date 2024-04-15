import numpy as np
import random
import matplotlib.pyplot as plt
class local_flags_FF:
	def __init__(self):
		self.test_flip_indices = False
		self.compare_prob_fl = True
		self.plot_k_states = False
	def compare_code(self):
		self.compare_prob_fl = True
flags_FF = local_flags_FF()
## band index is -1, 1
def momentum_state(bandindex, k, sigma, dim,n_cell, h, theta):
	k_states = np.zeros((4+dim, n_cell), dtype='complex128')
	k_states[0] += bandindex
	k_states[1:4] += compute_uvo(bandindex, k, sigma, dim, h, theta, n_cell)
	k_states[4:] += k.T
	return k_states
def create_all_k_states(N, dim,h, theta, d=1):
	if dim == 1:
		n_cell = int(N/2)
		k = [-np.pi/2 + 2*np.pi / (N)*i  for i in range (n_cell)]
		k = np.array(k)
	elif dim == 2:
		n_cell = int(N**2/2)
		lat_const =  1 #np.sqrt(2)
		shift = 1/(lat_const*N)
		factor = 2/(N*lat_const)
		start_values = [-1/lat_const+factor * i for i in range(N)]
		# get all k-states and substract the boundaries
		k = [[i, j] for i in start_values for j in start_values if
		 					((abs(i)+abs(j))<=1/lat_const and not np.isclose((abs(i)+j),1/lat_const))]
		k = np.array(k)
		k = k+shift
		#plt.plot(k[:, 0], k[:, 1], 'x' )
		#plt.plot([1, 0, -1, 0.5, 0.5, -0.5], [0, -1, 0, -0.5, 0.5, 0.5], 'x')
		#plt.show()
		k = np.pi*np.array(k)
		#comparison = np.array(start_values)
		#print(comparison, lat_const)
		#plt.plot(comparison, comparison, 'x')


		if len(k)!= n_cell:
			print('No')
	k_down1 = momentum_state(-1, k, sigma=-0.5,dim =dim,n_cell=n_cell, h=h, theta=theta)

	k_down2 = momentum_state(1, k, -0.5, dim, n_cell, h, theta)
	k_up1 = momentum_state(-1, k, 0.5, dim, n_cell,  h, theta)
	k_up2 = momentum_state(1, k, 0.5, dim, n_cell, h, theta)
	k_down = np.concatenate((k_down1, k_down2), axis=1)
	k_up = np.concatenate((k_up1, k_up2), axis=1)
	return np.array(k_down), np.array(k_up)
def compute_uvo(bandindex, k,sigma,dim, h, theta, n_cell):
	if dim==1:
		factor = 0.5*(np.exp(1j*np.pi*theta))
		Delta = factor*np.cos(k+ np.pi / (2 * n_cell))
	elif dim==2:
		factor = np.array([0.5*(np.exp(1j*np.pi*theta)),
		0.5*(np.exp(-1j*np.pi*theta))]).reshape((1,2))
		#print(factor)
		Delta = np.sum(factor*np.cos(k), axis=1) # (factor * np.cos(k))[:, 0] #
		#print(Delta)

	# Compute the energies
	omega = np.sqrt(abs(Delta)**2+h**2)
	omega = bandindex * omega
	if flags_FF.plot_k_states:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection = '3d')
		print(np.shape(k[:, 0]), np.shape(k[:, 1]), np.shape(omega))
		ax.scatter(k[:, 0], k[:, 1], omega)
		plt.show()

	# Set Delta where delta is zero to one, such that we have no instabilities when taking the fraction
	indices_zero = np.where(np.isclose(Delta, 0))[0]
	Delta[indices_zero] = 1
	add = sigma * h/omega if h !=0 else np.zeros(len(k))
	#print(np.shape(add), len(k))
	if bandindex==-1:
		u = np.conj(np.sqrt(0.5*(1+add)))
		v = np.conj(Delta/abs(Delta)*np.sqrt(0.5*(1-add)))
	elif bandindex==1:
		u = -Delta/abs(Delta)*np.sqrt(0.5*(1-add))
		v =  np.conj(np.sqrt(0.5*(1+add)))
	#print(np.shape(u), np.shape(v) , np.shape(omega))
	#print(Delta / abs(Delta))
	return u,v, omega
def update_row(new_value,index,k_state, matrix, inv, dets,  dim):

	if dim == 1:
		row1 = 1j * new_value *  k_state[4]
		row2 = 1j *new_value * (k_state[4]+np.pi )

	if dim == 2:
		new_value = np.array(new_value).reshape(2,1)
		row1 = 1j* np.sum( 1 * new_value *  k_state[4:], axis=0)
		row2 = 1j*np.sum(1* new_value * (k_state[4:]+np.pi), axis=0 )
		#print(np.shape(row1), np.shape(row2))
	new_row = 1/np.sqrt(2) * ((k_state[1]+k_state[2]) * np.exp(row1) + (k_state[1]-k_state[2]) *np.exp(row2))

	## Update the slater matrix
	slater_new = np.array(matrix)
    # Compute the determinant and inverse
	if dim ==1:
		det_new =  new_row.dot(inv[:, index])
		slater_new[index, :] = new_row
	elif dim ==2:
		det_new =  new_row.dot(inv[:, index])
		slater_new[index, :] = new_row#*10
	#det_new = np.linalg.det(slater_new.astype(complex))
	return(det_new, slater_new)
def spin_flip(sites, k_states, slater, invs, dets, dim):
	# For comparison
	# Get new value and index
	dim1 = len(sites[0])
	dim2 = len(sites[1])
	#For comparison of the code include extra artificial random step
	if flags_FF.compare_prob_fl:

		N_hole = 0
		n_spin_hole_choice = random.randint(1, dim1 + dim2)
		if n_spin_hole_choice < N_hole:
			random.randint(1, dim1 + dim2)

	index1 = np.random.randint(0,dim1)
	index2 = np.random.randint(0,dim2)
	if flags_FF.test_flip_indices:
		print(index1, index2)

	value1 = sites[0][index1]
	value2 = sites[1][index2]
	# Switch spins
	new_sites = np.array(sites)
	new_sites[0][index1] = value2
	new_sites[1][index2] = value1

	# Compute the new determinant, inv and slater
	det1, slater1 = update_row(value2, index1, k_states[0], slater[0], invs[0], dets[0], dim)
	det2,  slater2 = update_row(value1, index2, k_states[1], slater[1], invs[1], dets[1], dim)
	# create new data structures
	ndets = [det1, det2]
	nslater = [slater1, slater2]
	return (new_sites, nslater, ndets, [index1, index2], [value2, value1])
def update_column(index,new_k, sites, slater, inv, det, dim):
	#print(np.shape(sites))
	if dim == 1:
		col1 = 1j * new_k[4] *  sites
		col2 = 1j *(new_k[4]+np.pi) * sites
	if dim == 2:
		col1 = 1j* np.sum( new_k[4:] *  sites, axis=1)
		col2 = 1j*np.sum((new_k[4:] + np.pi) * sites, axis=1)
		#print(np.shape(col1), np.shape(col2))
	new_col = 1 / np.sqrt(2) * ((new_k[1]+new_k[2]) * np.exp(col1) + (new_k[1]-new_k[2]) *np.exp(col2))
	#print(np.shape(new_col))
	# Compute new column

	# Compute new slater

	slater_new = np.array(slater)
	slater_new[:, index] = new_col
	# Compute new dets
	#det_new=np.linalg.det(slater_new.astype(complex))

	det_new =  inv[index, :].dot(new_col.T)
	#det_new = np.linalg.det(slater_new.astype('complex'))
	#print(det_new, np.linalg.det(slater_new.astype('complex')))
	#if np.isclose(det_new, np.linalg.det(slater_new.astype('complex'))):
		#print('Yay!')
	#else:
		#print(det_new, np.linalg.det(slater_new))
		#sys.exit(0)

	return(det_new, slater_new)
def k_flip(sites, k_states, o_k_states, slater, invs, dets, dim):

	# Select randomply spin up or down.
	spin = random.randint(0, 1)
	#print(spin)
	# Select the index
	n_o_states = len(o_k_states[spin][0])
	n_states = len(k_states[spin][0])
	index_k = np.random.randint(0, n_states)
	index_o_k = np.random.randint(0, n_o_states)
	if flags_FF.test_flip_indices:
		print('Spin indices', spin, index_k, index_o_k)
	# # Artifically make random such that we get the same results
	if flags_FF.compare_prob_fl:
		random.randint(0,1)
	# create new k_state
	old_k=k_states[spin][:, index_k]
	new_k = o_k_states[spin][:, index_o_k]
	if flags_FF.test_flip_indices:
		print('new old', new_k, old_k)

        # compute the new column
	nk_states = [np.array(k_states[0]), np.array(k_states[1])]
	nk_states[spin][:, index_k] = new_k
	no_k_states = [np.array(o_k_states[0]), np.array(o_k_states[1])]
	no_k_states[spin][:, index_o_k] = old_k
	det1, slater1 = update_column(index_k, new_k, sites[spin],
					slater[spin], invs[spin], dets[spin], dim)
	ndets = np.array(dets, dtype='complex')
	#print(ndets)
	ndets[spin] = det1
	nslater = np.array(slater)
	nslater[spin] = slater1
	# Indices of change
	indices = [spin,index_k]
	return nk_states, no_k_states, nslater, ndets, indices, new_k
def update_det_inv(new_row, index, det, inv):
	factor = new_row.dot(inv[:, index])
	ndet = det * factor
	# An identity matrix is initialized.
	identity = np.eye(dim)

	# Matrix with a one at [proposed_index, proposed_index] and zero everywhere else is created.
	single_one_matrix = np.zeros((dim, dim))
	single_one_matrix[index, index] = 1

	# The matrix with everything equal to zero except for the changed row or column.
	value_matrix = np.zeros((dim, dim), dtype=np.longcomplex)

	# Difference between row and column change.

	# The inverse of the product of the changed row with the column of the current inverse.
	K_inv = np.float_power(factor, -1, dtype=np.longcomplex)

	# value_matrix for row change.
	value_matrix[index, 0:dim] = new_row

	# For row changes update_1 is set to the current inverse.
	update_1 = inv

	# For row changes update_2 is set to the following expression.
	update_2 = identity + K_inv * (single_one_matrix - value_matrix.dot(inv))
	inv_new = update_1.dot(update_2)
	return inv_new
