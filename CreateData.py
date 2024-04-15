import time
from pickle import load, dump
from MonteCarlo import *


# Create data from Variational Monte Carlo and save as piclöe file. Save the energies separately as txt file.
# Paramters: n_start=Number of starting conditions
#            N = 1D: number of sites; 2D: length of square
#            n_d = number of down spin electrons
#            n_up = number of up spin electrons
#            dim = 1D or 2D; the dimension of the sample
#            h, theta = variational parameters of the staggered flux+neel field
n_start, N, n_d, n_u = 3, 4, 8, 8
tmp = 0.2
dim = 2
h, theta = 0, 0.1


def create_samples(n_start, N, N_d, N_u,  T, dim, h, theta,  factor=1000):
    #Seed
    #np.random.seed(3)
    #random.seed(3)
    start=time.time()
    chains = []
    chains_e = []

    for i in range(n_start):

        printProgressBar (i, n_start, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r")
        #p, states, invs_matrices, dets, klist
        ps, values, energies = compute_data(N,N_d, N_u,factor,  T, dim, h, theta)
        keep_s  = values[100*N**dim:]
        keep_states =keep_s[::2*N**dim]+keep_s[N**dim+1::2*N**dim]
        chains = chains+keep_s
        keep_e  = energies[100*N**dim:]
        keep_e =energies[::2*N**dim]+ energies[N**dim+1::2*N**dim]
        chains_e = chains_e+keep_e


    end=time.time()
    return np.array(chains), [np.mean(chains_e), np.var(chains_e) ]

def compute_data(N,down, up, factor, T, dim, h, theta):
    start=time.time()
    a=Sample(N,[down, up], T, dim, h, theta)
    p, all_data, energies =a.monte_carlo(a.length**a.dim*factor, T  )
    end=time.time()
    return p, all_data, energies







start = time.time()

samples, energy = create_samples(n_start, N, n_d, n_u, tmp, dim, h, theta)



# Save the data

out = open('Data.pickle', 'wb')
dump(samples, out)
out.close()
np.savetxt( 'Data.txt', energy)
end = time.time()
print('Finished in ', end-start)
