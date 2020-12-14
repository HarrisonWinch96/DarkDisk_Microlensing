from plotgen_functions import *
import sys

print('started', sys.argv[1])

error = float(sys.argv[1])

print(type(error))

baryon_errors(error)

print('all done :) for ', format(error))