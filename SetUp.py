
import random
import numpy as np
from FlipFunctions import *
class local_flags_SU:
    def __init__(self):
        self.test_replace = False
flags_SU = local_flags_SU()
def compute_energy(N, sites, k_states, invs, dim):
    # Compute Energy of a state
    if dim == 1:
       energy = compute_energy_1D(N, sites, k_states, invs)
    if dim == 2:
       energy = compute_energy_2D(N, sites, k_states, invs)
    return energy
def compute_energy_1D(N, sites, k_states, invs,temperature = 0):
    # Compute the energy E_z and indices of switch
    sites = np.array(sites)
    state = np.array(N*[0])
    state[sites[0]] = -1
    state[sites[1]] = 1
    switched = np.concatenate([state[1:] * state[:-1], [state[0] * state[-1]]])
    E_z = (np.sum(switched)) / 4
    borders = np.where(switched == -1) [0]
    # For every switch compute the contribution
    prob_state = []
    list_spin = np.where(sites==0)
    spin = list_spin[0][0]

    n_bonds = len(borders)
    for position in borders:
        old_index = np.where(sites[spin]==position)[0][0]
        spin2 = (spin+1)%2
        new_index = np.where(sites[spin2]==(position+1) % N )[0][0]
       ## Compute new dets
        ndets1, _= update_row(position+1,old_index,k_states[spin],
                              invs[spin], invs[spin],
                              [1,1],  dim=1)
        ndets2, _= update_row(position, new_index,k_states[spin2],
                              invs[spin2], invs[spin2],
                              [1,1],  dim=1)
        det = ndets1*ndets2
        prob_state.append(det)
        # Change to new index
        spin = spin2
    E_xy= -1/2 * np.sum(prob_state) #/ n_bonds

    return [E_z / N, E_xy / N]
def compute_energy_2D(N, sites, k_states, invs,temperature = 0):
    # Compute the energy E_z and indices of switch
    state = np.zeros((N, N))
    state[sites[0][:, 0], sites[0][:, 1]] = -1
    state[sites[1][:, 0], sites[1][:, 1]] = 1
    switched_row = state[1:] * state[:-1]
    switched_col = state[:, 1:] * state[:, :-1]
    E_z = (np.sum(switched_col) + np.sum(switched_row) )  / 4
    borders_col = np.array(np.where(switched_col == -1) )
    borders_row = np.array(np.where(switched_row == -1))

    # For every switch compute the contribution
    prob_state = []
    # Every Row
    for position in borders_row.T:
        # Check which spin
        if (sites[0] == position).all(1).any():
            spin = 0
        elif (sites[1] == position).all(1).any():
            spin = 1
        # Check where
        old_index = (sites[spin] == position).all(1).argmax()
        neighbour = [position[0]+1, position[1]]
        spin2 = (spin+1)%2
        new_index = (sites[spin2]==neighbour).all(1).argmax()
       ## Compute new dets
        ndets1, _= update_row(neighbour,old_index,k_states[spin],
                              invs[spin], invs[spin],
                              [1,1],  dim=2)
        ndets2, _= update_row(position, new_index,k_states[spin2],
                              invs[spin2], invs[spin2],
                              [1,1],  dim=2)
        det = ndets1*ndets2
        prob_state.append(det)
    # Columns
    for position in borders_col.T:
        # Check which spin
        if (sites[0] == position).all(1).any():
            spin = 0
        elif (sites[1] == position).all(1).any():
            spin = 1
        # Check where
        old_index = (sites[spin] == position).all(1).argmax()
        neighbour = [position[0], position[1]+1]
        spin2 = (spin+1)%2
        new_index = (sites[spin2]==neighbour).all(1).argmax()
       ## Compute new dets
        ndets1, _= update_row(neighbour,old_index,k_states[spin],
                              invs[spin], invs[spin],
                              [1,1],  dim=2)
        ndets2, _= update_row(position, new_index,k_states[spin2],
                              invs[spin2], invs[spin2],
                              [1,1],  dim=2)
        det = ndets1*ndets2
        prob_state.append(det)

    E_xy = -1/2 * np.sum(prob_state)
    #print(E_xy, E_z)
    if (abs(E_xy+E_z) > 1.5*N**2):
        pass
        #print(det, E_xy, E_z)
    energy = [E_z/(N*(N-1)), E_xy/(N*(N-1))]
    return energy
def get_initial_k(length, spin_number, temperature, dim, h, theta, lat_const=1):
    '''computes the first sample k_state depending on finitie or zero temperature
       Parameters:
            N: int Number of lattice sites
            spin_number: list of int spin_down spin_up
            temperature: float, temperature
            lat_const:   float, lattice constant
        Returns:
            [k_down, k_up]: list of arrays with the sample k_states
  	        [o_k_down, o_k_up]: list arrays the k_states not used
    '''

    k_states_down, k_states_up = create_all_k_states(N = length, dim= dim, d=lat_const, h=h, theta=theta)
    N=length**dim
    if temperature != 0:
        k_down_indices = random.sample(range(N), spin_number[0])
        k_up_indices = random.sample(range(N), spin_number[1])
        o_k_down_indices = list(set(range(N)).
                                difference(set(k_down_indices)))
        o_k_up_indices = list(set(range(N)).difference(set(k_up_indices)))
        k_down = k_states_down[:, k_down_indices]
        k_up = k_states_up[:, k_up_indices]
        o_k_down = k_states_down[:, o_k_down_indices]
        o_k_up = k_states_up[:, o_k_up_indices]
    else:

        k_down = k_states_down[:, :spin_number[0]]
        k_up = k_states_up[:, :spin_number[1]]
        o_k_down = k_states_down[:, spin_number[0]:]
        o_k_up = k_states_up[:, spin_number[1]:]
    return [k_down, k_up], [o_k_down, o_k_up]
def get_initial_s(N, N_spin,dim):
    if dim == 1:
        all_sites = range(N)
    elif dim ==2:
        all_sites = [[x,y] for x in range(N) for y in range(N)]

    state = random.sample(all_sites, sum(N_spin))
    s_down = np.array(state[:N_spin[0]])
    s_up = np.array(state[N_spin[0]:])

    return [s_down, s_up]
# formula taken from Piazza doctoral thesis p. 50
def compute_weight(k_states, temperature):
    if temperature == 0:
        return 1
    else:
        energy = sum(k_states[3])
        #print('Energy', energy)
        return np.exp(-1/temperature*energy)
def compute_probability(k_states, dets, temperature):
    p_b = (compute_weight(k_states[0], temperature=temperature)
         * compute_weight(k_states[1], temperature=temperature))
    p = p_b * abs(dets[0] * dets[1])** 2
    return p
def monte_carlo_step(data, p_k, N, temperature, dim,lat_const, mode = None):
    # Declare indices
    indices_s = None
    indices_k = None
    # unpack data
    sites, k_states, o_k_states, dets, slater, invs, p = data
    # For T=0 set probability of k_flip to zero
    if temperature == 0:
        p_k = 0
    # Decide on spin or k flip
    choice = random.random()
    if choice > p_k:
        nsites, nslater, ndets, indices_s, new_value = spin_flip(sites, k_states,
                                            slater, invs, dets, dim)
        nk_states = k_states
        no_k_states = o_k_states
        p_new = compute_probability(k_states, ndets, temperature)
    else:
        nk_states, no_k_states, nslater, ndets, indices_k, new_value = k_flip(sites,
                            k_states, o_k_states,
                            slater, invs, [1,1], dim)
        nsites = sites
        p_new = compute_probability(nk_states, ndets, temperature)
	# Compare the probabilites
    p_new = p_new / compute_probability(k_states, [1,1], temperature)
    # Decide on whether to accept
    replace = False
    # accept with probability of ratio
    draw = random.random()
    if draw < p_new:
        replace = True
    if flags_SU.test_replace:
        print(replace, 'p_new', p_new, 'random_number', draw)
    if replace:
        ninvs = [np.linalg.inv(matrix.astype('complex')) for matrix in nslater]
        if mode == 'energy':

            energy = compute_energy(N, nsites, nk_states, ninvs, dim)
            return [nsites, nk_states, no_k_states, ndets, nslater, ninvs, p_new], energy
        if mode == 'entropy':
            indices = [indices_s, indices_k]
            return [nsites, nk_states, no_k_states, ndets, nslater, ninvs, p_new], indices, new_value
    else:
        if mode == 'energy':
            return [sites, k_states, o_k_states, dets, slater, invs, 1], None
        if mode == 'entropy':
                return [sites, k_states, o_k_states, dets, slater, invs, 1], None, None
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
