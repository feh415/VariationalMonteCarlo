rom getData import *
from pickle import load, dump



# Create data from Variational Monte Carlo and save as piclöe file. Save the energies separately as txt file.
# Paramters: n_start=Number of starting conditions
#            N = 1D: number of sites; 2D: length of square
#            n_d = number of down spin electrons
#            n_up = number of up spin electrons
#            dim = 1D or 2D; the dimension of the sample
#            h, theta = variational parameters of the staggered flux+neel field
n_start, N, n_d, n_u = 2, 3, 8, 8
tmp = 0.2
dim = 2
h, theta = 0, 0.1
samples, energy = create_samples(n_start, N, n_d, n_u, tmp, dim, h, theta)
print('Total energy = ',  energy)
out = open(file_name+'.pickle', 'wb')
dump(samples, out)
out.close()
np.savetxt( file_name+'.txt', energy)
end = time.time()
print('Finished in ', end-start)
