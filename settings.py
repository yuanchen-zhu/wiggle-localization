EPS = 1e-14

# Number of vertices of the initiallly generated graph. The actual
# graph used for testing however will be pruned further depending on
# the various parameters.
PARAM_V = 200

# Dimesion of graph configuration / embedding.
PARAM_D = 2

# List of std. dev noises to test
PARAM_NOISE_STD = 0

# List of perturb-to-noise ratios to test
PARAM_PERTURB = 10

# The number of perturbed L measurement samples to be taken is the
# dimension of the actual tangent space times numbers in this
# list. Each entry in the list will be individually tested upon.
PARAM_SAMPLINGS = 16

# Threshold controlling the longest edge in the generated graph.
PARAM_DIST_THRESHOLD = 4.5

# Minimal perturbation standard deviation
PARAM_MIN_PERTURB =  1e-4

# Max vertex degree of generated graph.
MAX_DEGREE = 10

# Whether to prune the generated graph so only one globally linked
# component is left.
SINGLE_LC = False

# Whether to filter out dangling vertices and components. The
# filtering is simpe minded, but increases likely hood of the graph
# being locally rigid.
FILTER_DANGLING = False

# The length of the longest edge in the graph measured in meters. This
# is only useful for printing testing results in real-world units.
MAX_EDGE_LEN_IN_METER = 30.0

# Ratio of 1 meter / 1 unit length in the test. This global variable
# is set in dist.main, calculated from MAX_EDGE_LEN_IN_METER
METER_RATIO = 0

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

# When STRESS_SAMPLE = semilocal | local, each vertex together with
# its K_RING-ring is considered as the starting point of each local
# subgraph. More rings are added to the subgraph till at least
# MIN_LOCAL_NBHD vertices are contained in it.
K_RING = 2
MIN_LOCAL_NBHD = 30

# When STRESS_SAMPLE = semilocal, after running PCA on basis of all
# local stress spaces, the left singular vectors corresponding to
# singular values less than STRESS_VAL_PERC * median(singular values)
# are discarded.
STRESS_VAL_PERC = 0

# Whether to use sparse svd or dense svd to consolidate local stress
# spaces into the global stress space (requires the 3rd party svd
# executable). Only used if STRESS_SMPLE == 'semilocal'
USE_SPARSE_SVD = True

# The number of stress kernel samples taken from each stress matrix
# sample equals max(v * KERNEL_SAMPLE_RATIO, KERNEL_SAMPLE * dim K(p)
KERNEL_SAMPLES = 1.2
KERNEL_MAX_RATIO = 1

# Whether the set stress vector samples (source of stress matrix)
# should be made orthonormal. Don'
ORTHO_SAMPLES = True

# Whether measurement noise is multiplicative or not
MULT_NOISE = False

# Whether exact local stress spaces should be calculated. Currently
# doesn't work.
EXACT_LOCAL_STRESS = False

# Number of stress space samples, i.e., the number of stress matrix
# samples.
SS_SAMPLES = 20

# Whether the stress kernel samples taken from each stress matrix
# sample should be weighted based on their correponding singular
# values. Almost always makes result worse.
WEIGHT_KERNEL_SAMPLE = False

# Whether pca consolidation of local stress space checks the disk-cache first
CONSOLIDATE_READ_FROM_CACHE = True

# Wheter the result of stress space consolidation is written to the
# disk-cache.
CONSOLIDATE_WRITE_TO_CACHE = True

# Use svd instead of eig to calculate stress kernels
SK_USE_SVD = False

# Whether consolidation of stress kernel samples using PCA happens
# globally, or for each globally linked component using the
# corresponding portion of the stress kernel vectors.
PER_LC_KS_PCA = False

# The number of coordinate vectors used to reconstruct the
# configuration equals this number times the actual dimension of the
# configuration. If 0, then the least square solver will be used,
# using d coordinate vectors.
SDP_SAMPLE = 4

# Use the faster DSDP5 solver for solving semi-definite programming
SDP_USE_DSDP = True

# Whether the configuration obtained through affine optimization is
# plotted
PLOT_AFFINE_FIT = False

# If non-null, specifies the floor plan file used for testing. Must be
# in MITquest XML format.
FLOOR_PLAN_FN = "space.xml?10-2"

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
