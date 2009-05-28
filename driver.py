#!/usr/bin/python

import settings as S
import dist
import random
import copy
import sys
from multiprocessing import Pool


def simulation_params():
    params = S.SimParams()
    RANDOM_SEED = 100                     # Self explanatory

    params.V = 200
    params.D = 2
    params.MULT_NOISE = False
    params.DIST_THRESHOLD = 8
    params.MAX_EDGE_LEN_IN_METER = 30.0
    params.SINGLE_LC = False
    params.EXACT_STRESS = False
    params.TRILATERATION = False
    params.SAMPLINGS = 4
    params.MIN_PERTURB =  1e-12
    params.SS_SAMPLES = 200
    params.SDP_SAMPLE_MIN = 1
    params.SDP_SAMPLE_MAX = 5
    params.SDP_SAMPLE_ENUMERATE_THRESHOLD = 20
    params.K_RING = 2
    params.MIN_LOCAL_NBHD = 20
    params.STRESS_VAL_PERC = 2
    params.RANDOM_STRESS = True
    params.ORTHO_SAMPLES = True

    rnd = random.Random()
    rnd.seed(10)
    seeds = [ rnd.randint(0,65536) for i in xrange(20) ]

    output_params = S.OutputParams()

    for params.STRESS_SAMPLE in ['global', 'semilocal']:
        for params.FLOOR_PLAN_FN in [None, "space.xml?10-2"]:    
            for params.MAX_DEGREE in xrange(6, 13):
                    for params.NOISE_STD, params.PERTURB in [(1e-3, 20), (1e-4, 80)]:
                        for params.RANDOM_SEED in seeds:
                            yield (copy.deepcopy(params), output_params)

def simulate_wrapper(item):
#    print item[0], item[1]
    return dist.simulate(item[0], item[1], False)

def go(num_worker):
    pool = Pool(num_worker)
    pool.map(simulate_wrapper, simulation_params(), 1)
    
if __name__ == "__main__":
    num_workers = 4
    if len(sys.argv) > 1:
        num_workers = int(sys.argv[1])
    if num_workers > 1:
        go(num_workers)
    else:
        map(simulate_wrapper, simulation_params())
