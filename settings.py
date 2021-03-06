######################################################################
# Output control
######################################################################
class OutputParams:
    PLOT_SIZE = 5
    MARGIN = 0.06
    SKIP_PLOT = False
    ENUMERATE_GLC = False # Only enumerate generic linked component without running
    DUMP_LC = False       # Dump subgraphs
    PLOT_POINT_SIZE = 10
    SKIP_PRINT_OLD = True
    CONVERT_PDF = True
    OPEN_EPS = False
    GENERATE_TXT_GRAPH = False # Only dump generated random graphs without running

######################################################################
# Default simulation parameters controlling input fed to the simulation
######################################################################

class SimParams:
    RANDOM_SEED = 1                     # Self explanatory

    # If non-null, specifies the floor plan file used for
    # testing. Must be in MITquest XML format.
    
    FLOOR_PLAN_FN = "space.xml"

    # Number of vertices of the initiallly generated graph. The actual
    # graph used for testing however will be pruned further depending on
    # the various parameters.
    V = 200
    
    # Dimesion of graph configuration / embedding.
    D = 2
    
    # std. dev noises to test
    NOISE_STD = 1e-4 #1e-3
    
    # Whether measurement noise is multiplicative or not
    MULT_NOISE = False
    
    # Threshold controlling the longest edge in the generated graph.
    DIST_THRESHOLD = 8
    #DIST_THRESHOLD = 1
    
    # The length of the longest edge in the graph measured in meters. This
    # is only useful for printing testing results in real-world units.
    MAX_EDGE_LEN_IN_METER = 30.0
    
    # Ratio of 1 meter / 1 unit length in the test. This global variable
    # is calculated from MAX_EDGE_LEN_IN_METER
    
    def dist_threshold(self):
        import math
        k =  math.pow(2*(self.D+1), 1.0/self.D)/3.0
        return self.DIST_THRESHOLD * k * math.pow(self.V, -1.0/self.D)
    
    def meter_ratio(self):
        return self.MAX_EDGE_LEN_IN_METER / self.dist_threshold()
    
    # Max vertex degree of generated graph.
    MAX_DEGREE =  8
    
    # Whether to prune the generated graph so only one globally linked
    # component is left.
    SINGLE_LC = False

    ######################################################################
    # Algorithm parameters controlling execution of the algorithm
    ######################################################################
    
    # Method to calculate stress space: 'global' | 'semilocal' |
    # 'local'. If equals 'global', the tangent space of the entire graph
    # is calculated and the stress space is set to be its orthogonal
    # complement. If equals 'semilocal', the stress space of subgraphs are
    # calculated using the 'global' method, and then these sub stress
    # spaces are pieced together using PCA. If equals 'local', the stress
    # space of subgraphs are calculated as in 'semilocal,' but the stress
    # samples are then calculated as sums of random samples from each
    # local stress space, i.e., the global stress space is never
    # calculated.
    STRESS_SAMPLE = 'global'
    
    # Whether to use exact stress space from coordinates
    EXACT_STRESS = False
    
    # Whether to trilaterate
    TRILATERATION = False
    
    # perturb-to-noise ratios to test
    # Set this to 80 for low noise (1e-4) and 20 for high noise (1e-3)
    PERTURB = 20
    
    # The number of perturbed L measurement samples to be taken is the
    # dimension of the actual tangent space times numbers in this
    # list. Each entry in the list will be individually tested upon.
    SAMPLINGS = 8
    
    # Minimal perturbation standard deviation
    MIN_PERTURB =  1e-12
    
    # Number of stress space samples, i.e., the number of stress matrix
    # samples.
    SS_SAMPLES = 200
    
    # The number of coordinate vectors used to reconstruct the
    # configuration equals this number times the actual dimension of the
    # configuration. If MAX=0, then the least square solver will be used,
    # using d coordinate vectors.
    SDP_SAMPLE_MIN = 1
    SDP_SAMPLE_MAX = 5
    SDP_SAMPLE_ENUMERATE_THRESHOLD = 20
    SDP_SAMPLE_MAX_ENUMERATE = 4
    
    
    # When STRESS_SAMPLE = semilocal | local, each vertex together with
    # its K_RING-ring is considered as the starting point of each local
    # subgraph. More rings are added to the subgraph till at least
    # MIN_LOCAL_NBHD vertices are contained in it.
    
    K_RING = 2
    MIN_LOCAL_NBHD = 20
    
    # When STRESS_SAMPLE = semilocal, after running PCA on basis of all
    # local stress spaces, the left singular vectors corresponding to
    # singular values less than STRESS_VAL_PERC * median(singular values)
    # are discarded.
    STRESS_VAL_PERC = 2
    
    # Whether to create random stress or to use the calculated basis of
    # the stress space
    RANDOM_STRESS = True
    
    # if random stress is used, whether the set stress vector samples
    # (source of stress matrix) should be made orthonormal.
    ORTHO_SAMPLES = True

######################################################################
# General program settings. These rarely need to be changed.
######################################################################

EPS = 1e-14

# Whether pca consolidation of local stress space checks the disk-cache first
CONSOLIDATE_READ_FROM_CACHE = False

# Wheter the result of stress space consolidation is written to the
# disk-cache.
CONSOLIDATE_WRITE_TO_CACHE = False

# Directory where plot files reside
DIR_PLOT = "plot"

# Directory where source data resides
DIR_DATA = "data"

# Directory where 1st/3rd party binaries reside
DIR_BIN = "bin"

# Directory where temporary files reside
DIR_TMP = "tmp"

# Directroy where cache file reside
DIR_CACHE = "cache"

# Use the faster DSDP5 solver for solving semi-definite programming
SDP_USE_DSDP = False

# Whether to use sparse svd or dense svd to consolidate local stress
# spaces into the global stress space (requires the 3rd party svd
# executable). Only used if STRESS_SMPLE == 'semilocal'
USE_SPARSE_SVD = True
