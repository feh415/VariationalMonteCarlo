import numpy as np
from SetUp import*

class local_flags_MC:
    def __init__(self):
        self._results = False
        self.plot_results = False
         # 0 for comparing s_state, 1 for comparing k_staten None else
        self.compare_omc = None
        ## Aritificially increase number of random events taken to compare the codes
        self.compare_prob_omc = None # 0, 1, 2, 3 for a_k, b_k, a_p, b_p
    def compare_code(self, slater_id):
        self.compare_omc = None
        if slater_id =='a_k':
            self.compare_prob_omc = 0
        elif slater_id == 'a_p':
            self.compare_prob_omc = 2
        elif slater_id == 'b_p':
            self.compare_prob_omc = 3
        np.random.seed(10)
        random.seed(10)
flags_MC = local_flags_MC()
class Sample:
    def __init__(self, N, N_spin, temperature, dim, h, theta):
        # Seeds
        #np.random.seed(0)
        #random.seed(0)
        self.length = N
        self.dim=dim
        self.spin_number = N_spin
        self.k_state, self.o_k_state = get_initial_k(N, N_spin, temperature, dim, h, theta)
        self.s_state = get_initial_s(N, N_spin, dim)
        ## Make 3 more times such that we have the same random
        ## a, k extracted from a_k
        ## b_p extracted from b_p
        if flags_MC.compare_prob_omc == 3:
            k, o_k = get_initial_k(N, N_spin, temperature, dim, h, theta)
            s = get_initial_s(N, N_spin, dim)
            k, o_k = get_initial_k(N, N_spin, temperature, dim, h, theta)
            s = get_initial_s(N, N_spin, dim)
            self.k_state, self.o_k_state = get_initial_k(N, N_spin, temperature, dim, h, theta)
            self.s_state = get_initial_s(N, N_spin, dim)
        elif flags_MC.compare_prob_omc == 0:
            print('This is correct')
            for i in range(3):
                k, o_k = get_initial_k(N, N_spin, temperature, dim, h, theta)
                s = get_initial_s(N, N_spin, dim)
        elif flags_MC.compare_prob_omc == 2:
            k, o_k = get_initial_k(N, N_spin, temperature, dim, h, theta)
            s = get_initial_s(N, N_spin, dim)
            k, o_k = get_initial_k(N, N_spin, temperature, dim, h, theta)
            s = get_initial_s(N, N_spin, dim)
            self.k_state, self.o_k_state = get_initial_k(N, N_spin, temperature, dim, h, theta)
            s = get_initial_s(N, N_spin, dim)



    def compute_slater(self):
        slater = []
        for spin in range(len(self.spin_number)):
            k=self.k_state[spin]
            sites = self.s_state[spin]
            factor_1= 1/np.sqrt(2)* (k[1]+k[2])
            factor_2 = 1/np.sqrt(2)* (k[1]-k[2])
            if self.dim == 1:
                 exponents_1 = [1j*k[4]*site for site in sites]
                 exponents_2 = [1j*(k[4]+np.pi)*site for site in sites]
            if self.dim ==2:
                exponents_1 = [1*1j*np.sum(k[4:]*site.reshape(2,1), axis=0) for site in sites]
                exponents_2 = [1*1j*np.sum((k[4:]+np.pi)*site.reshape(2,1), axis = 0) for site in sites]
            slater_spin = factor_1 * np.exp(exponents_1) + factor_2 *np.exp(exponents_2)
            slater.append(np.array(slater_spin))
        return slater
    def compute_inv(self, slater):
        return [np.linalg.inv(slater[0].astype('complex')),
				            np.linalg.inv(slater[1].astype('complex'))]
    def compute_det(self):
        #matrices = self.compute_slater()
       # dets = []
        #for matrix in matrices:
            #[sgn, log_det] = np.linalg.slogdet(matrix.astype(complex))
            #det = sgn*np.exp(log_det)
            #dets.append(det)
        #if self.dim ==2:
        dets = [1, 1]
        return dets
    def get_probability(self, temperature):
        boltzmann_weight = compute_weight(self.k_state[0], temperature)
        boltzmann_weight *= compute_weight(self.k_state[1], temperature)
        dets = self.compute_det()
        #if self.dim == 2:
        p = boltzmann_weight

        #else:
             #p = abs( dets[1]*dets[0]) ** 2 *boltzmann_weight

        return p
    def monte_carlo(self, steps, temperature, return_states = False, p_k=0.3):

        if flags_MC.compare_prob_omc != None :
            random.seed(10)
            np.random.seed(10)
        # Counter and return arrays

        ps = []
        all_data = []
        mean_energies = []
        # Initialize the data
        sites = self.s_state
        k_states = self.k_state
        o_k_states = self.o_k_state
        dets = self.compute_det()
        slater = self.compute_slater()
        invs = self.compute_inv(slater)

        p = self.get_probability(temperature)
        N = self.length
        dim=self.dim
        data = sites, k_states, o_k_states, dets, slater, invs, p # slows down !
        states = []
        energies = [0]
        energies_xy = [0]
        if p == 0:
            print ('error', p)
        ps.append(1)
        # Perform MonteCarlo steps
        failed = 0
        for i in range(steps):
            # OneMonteCarloStep
            #print(i)
            if i%(int(steps / 4)) == 0:
                print(i)
            data, energy = monte_carlo_step(data, p_k, N, temperature, dim, lat_const=1, mode = 'energy')
            # Compute energies
            mean_energies.append(data[1])
            if energy:
                energies.append(energy[0])
                energies_xy.append(energy[1])
            else:
                energies.append(energies[-1])
                energies_xy.append(energies_xy[-1])
            # Get the states

            ps.append(data[6]*ps[-1])
            if return_states:
                down_sites = np.array(data[0][0])
                up_sites = np.array( data[0][1])
                if dim == 1:
                    state = np.array(N* [0])
                    state[down_sites] = -1
                    state[up_sites] = 1

                if dim == 2:
                    state = np.zeros((N, N))
                    state[down_sites[:, 0], down_sites[:, 1]] = -1
                    state[up_sites[:, 0 ], up_sites[:, 1]] = 1
                states.append(state)

        total_energies = np.array(energies) + np.array(energies_xy)
        if flags_MC.plot_results:
            plt.yscale('log')
            plt.plot(ps)
            plt.show()
        if flags_MC.compare_omc == 1:
            print(np.shape(p))
            return ps, data[1], data[2]

        elif flags_MC.compare_omc == 0:
            return ps, data[0]
        return ps, states, list(total_energies)
