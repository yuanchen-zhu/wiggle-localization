#!/usr/bin/python

# Ubuntu Hardy annoyance: the XML lib I use is deprecated
import sys
sys.path = ['/usr/lib/python%s/site-packages/oldxml' % sys.version[:3]] + sys.path 

import settings as S
import dist

gtype = [ None, None, "space.xml?10-2", "space.xml?10-2"]
gname = ["iso-high", "iso-low", "aniso-high", "aniso-low"]
gdeg = [7, 7, 9, 9]
gsdps = [2, 2, 3, 3]
perturbs = [20, 80, 20, 80]
noise = [1e-3, 1e-4, 1e-3, 1e-4]
sss = [200, 200, 200, 200]
samplings = [4, 4, 4, 4]
titles = ["isotropic, high noise", "isotropic, low noise", "anisotropic, high noise", "anisotropic, low noise"]

def assign_S(i):
    global S
    S.FILTER_DANGLING =  True
    S.FLOOR_PLAN_FN =  None
    S.K_RING =  2
    S.MAX_DEGREE =  7
    S.MIN_LOCAL_NBHD =  20
    S.MULT_NOISE =  False
    S.ORTHO_SAMPLES =  True
    S.PARAM_D =  2
    S.PARAM_DIST_THRESHOLD =  5
    S.PARAM_MIN_PERTURB =  1e-06
    S.PARAM_NOISE_STD =  0.0001
    S.PARAM_PERTURB =  80
    S.PARAM_SAMPLINGS =  4
    S.PARAM_V =  200
    S.RANDOM_SEED =  0
    S.RANDOM_STRESS =  True
    S.SDP_SAMPLE_RATIO = 0.5
    S.SDP_SAMPLE_MIN = 1
    S.SDP_SAMPLE_MAX =  2
    S.SDP_USE_DSDP =  True
    S.SINGLE_LC =  True
    S.SS_SAMPLES =  200
    S.STRESS_SAMPLE =  'global'
    S.STRESS_VAL_PERC =  0
    S.TRILATERATION =  False
    S.USE_SPARSE_SVD =  True
    
    S.FLOOR_PLAN_FN = gtype[i]
    S.MAX_DEGREE = gdeg[i]
    S.SS_SAMPLES = sss[i]
    S.SDP_SAMPLE_MAX = S.SDP_SAMPLE_MIN = gsdps[i]
    S.SDP_SAMPLE_RATIO = 0.5
    S.PARAM_PERTURB = perturbs[i]
    S.PARAM_NOISE_STD = noise[i]
    S.PARAM_SAMPLINGS = samplings[i]


def tri():
    global S
    assign_S(1)
    S.TRILATERATION= True
    dist.simulate()
    assign_S(3)
    S.TRILATERATION = True
    dist.simulate()

def draw():
    global S
    for i in xrange(4):
        assign_S(i)
        dist.simulate()
    
def vary_mu():
    global S
    
    gp = open('varying-mu.gp', 'w')
    gp.write('plot \\\n')
    for i in xrange(4):
        assign_S(i)

        fn = "%s/varying-mu-%s.plot" % (S.DIR_PLOT, gname[i])
        if i != 0:
            gp.write(", \\\n")
        gp.write('\'%s\' using 1:2 lw 3 t "%s"' % (fn, titles[i]))
        gp.flush()

        f = open(fn, 'w')
        for S.PARAM_SAMPLINGS in [1, 2, 4, 8, 16, 32]:
            error_p, error_d = dist.simulate()
            f.write("%g %g %g %g %g\n" % (S.PARAM_SAMPLINGS, error_p, error_d, error_p * S.meter_ratio(), error_d * S.meter_ratio()))
            f.flush()
        f.close()
    gp.close()


def vary_N():
    global S
    gp = open('varying-N.gp', 'w')
    gp.write('plot \\\n')

    for i in xrange(4):
        assign_S(i)
        fn = '%s/varying-N-%s.plot' % (S.DIR_PLOT, gname[i])
        if i != 0:
            gp.write(", \\\n")
        gp.write('\'%s\' using 1:2 lw 3 t "%s"' % (fn, titles[i]))
        gp.flush()

        f = open(fn, 'w')
        for S.SS_SAMPLES in [1, 2, 10, 25, 50, 100, 200]:
            error_p, error_d = dist.simulate()
            f.write("%g %g %g %g %g\n" % (S.SS_SAMPLES, error_p, error_d, error_p * S.meter_ratio(), error_d * S.meter_ratio()))
            f.flush()
        f.close()
    gp.close()

def vary_delta():
    global S
    gp = open('varying-delta.gp', 'w')
    gp.write('plot \\\n')

    for i in xrange(4):
        assign_S(i)
        fn = '%s/varying-delta-%s.plot' % (S.DIR_PLOT, gname[i])
        if i != 0:
            gp.write(", \\\n")
        gp.write('\'%s\' using 1:2 lw 3 t "%s"' % (fn, titles[i]))
        gp.flush()

        f = open(fn, 'w')
        for S.PARAM_PERTURB in [5, 10, 20, 40, 80, 160,320]:
            error_p, error_d = dist.simulate()
            f.write("%g %g %g %g %g\n" % (S.PARAM_PERTURB, error_p, error_d, error_p * S.meter_ratio(), error_d * S.meter_ratio()))
            f.flush()
        f.close()
    gp.close()


def vary_D():
    global S
    gp = open('varying-D.gp', 'wb')
    gp.write('plot \\\n')

    for i in xrange(4):
        assign_S(i)
        fn = '%s/varying-D-%s.plot' % (S.DIR_PLOT, gname[i])
        if i != 0:
            gp.write(', \\\n')
        gp.write('\'%s\' using 1:2 lw 3 t "%s"' % (fn, titles[i]))
        gp.flush()

        f = open(fn, 'w')
        for S.SDP_SAMPLE_MAX in [1,2,3,4,5,6,7]:
            S.SDP_SAMPLE_MIN = S.SDP_SAMPLE_MAX
            error_p, error_d = dist.simulate()
            f.write("%g %g %g %g %g\n" % (int(S.SDP_SAMPLE_MAX * 2), error_p, error_d, error_p * S.meter_ratio(), error_d * S.meter_ratio()))
            f.flush()
        f.close()
    gp.close()

def enumerate_glc_non_ggr(i):
    global S
    for S.RANDOM_SEED in xrange(i*100, (i+1)*100):
        dist.simulate()

def e1():
    global S
    S.MAX_DEGREE=9
    enumerate_glc_non_ggr(0)

def e2():
    global S
    S.MAX_DEGREE=10
    enumerate_glc_non_ggr(1)

def e3():
    global S
    S.MAX_DEGREE=11
    enumerate_glc_non_ggr(2)


def e4():
    global S
    S.MAX_DEGREE=10
    enumerate_glc_non_ggr(4)

def different_sdp():
    global S
    for S.SDP_SAMPLE_MAX in [1, 2, 3, 4]:
        S.SDP_SAMPLE_MIN = S.SDP_SAMPLE_MAX
        dist.simulate()

def varying_v():
    global S

    S.FILTER_DANGLING =  True
    S.FLOOR_PLAN_FN =  None
    S.K_RING =  1
    S.MAX_DEGREE =  7
    S.MAX_EDGE_LEN_IN_METER =  30.0
    S.MIN_LOCAL_NBHD =  20
    S.MULT_NOISE =  False
    S.ORTHO_SAMPLES =  True
    S.PARAM_D =  2
    S.PARAM_DIST_THRESHOLD =  5
    S.PARAM_MIN_PERTURB =  1e-06
    S.PARAM_NOISE_STD =  0.0001
    S.PARAM_PERTURB =  80
    S.PARAM_SAMPLINGS =  4
    S.PARAM_V =  200
    S.RANDOM_SEED =  0
    S.RANDOM_STRESS =  True
    S.SDP_SAMPLE_MAX = S.SDP_SAMPLE_MIN =  2
    S.SDP_SAMPLE_RATIO = 0.5
    S.SDP_USE_DSDP =  True
    S.SINGLE_LC =  True
    S.SS_SAMPLES =  200
    S.STRESS_SAMPLE =  "semilocal"
    S.STRESS_VAL_PERC =  0
    S.TRILATERATION =  False
    S.USE_SPARSE_SVD =  True
    

    global dist
    
    gp = open('varying-v.gp', 'wb')
    fn = '%s/varying-v-iso-semi' % (S.DIR_PLOT)
    gp.write('plot \'%s\' using 1:2 lw 3 t ""' % fn)
    f = open(fn, 'wb')

    sdps = [2, 2, 2, 2, 3, 3, 3]
    i = 0
    for S.PARAM_V in [50, 70, 100, 141, 200, 283, 400]:
        S.SDP_SAMPLE_MAX = S.SDP_SAMPLE_MIN =sdps[i]
        error_p, error_d = dist.simulate()
        f.write("%g %g %g %g %g %g\n" % (S.PARAM_V, error_p, error_d, error_p * S.meter_ratio(), error_d * S.meter_ratio(), dist.NS))
        f.flush()
        i = i + 1
    f.close()
    gp.close()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        dist.simulate()
    else:
        i = int(sys.argv[1])
        d = [vary_mu, vary_N, vary_delta, vary_D, e1, e2, e3, e4, different_sdp, varying_v, tri, draw]
        d[i-1]()
        
        



