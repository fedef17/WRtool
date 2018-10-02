import itertools
import math

import numpy as np
from numpy import linalg as LA

# import cdms2
# cdms2.setNetcdfShuffleFlag(0)
# cdms2.setNetcdfDeflateFlag(0)
# cdms2.setNetcdfDeflateLevelFlag(0)


class MatchError(Exception):
    pass


class ConstantIndices(object):
    """
    A generator object designed to yield the same tuple of array indices
    as many times as needed.

    """

    def __init__(self, patterns):
        """Initialize the generator.

        Argument:
        patterns -- Integer number of patterns that are being matched.
                The number of indices is determined by this parameter
                and the number of times this generator will yield is
                determined by the factorial of the number of patterns.

        """
        self.total = math.factorial(patterns)
        self.value = tuple(range(patterns))
        self.count = 0

    def __iter__(self):
        """Just return the object when iter is called on it."""
        return self

    def next(self):
        """
        Yield the tuple of indices until the maximum number required is
        reached.

        """
        if self.count < self.total:
            self.count += 1
            return self.value
        else:
            raise StopIteration()


def pc_indices(n):
    """Return two generators to generate all of the permutaions."""
    c = ConstantIndices(n)
    v = itertools.permutations(range(n))
    n = c.total
    return c, v, n


def cosine(x,y):
    """
    Calculates the cosine of the angle between x and y. If x and y are 2D, the scalar product is taken using the np.vdot() function.
    """
    if x.ndim != y.ndim:
        raise ValueError('x and y have different dimension')
    elif x.shape != y.shape:
        raise ValueError('x and y have different shapes')

    if x.ndim == 1:
        return np.dot(x,y)/(LA.norm(x)*LA.norm(y))
    elif x.ndim == 2:
        return np.vdot(x,y)/(LA.norm(x)*LA.norm(y))
    else:
        raise ValueError('Too many dimensions')


def rms_calc_fullspace(clus_1, clus_2):
    """
    Computes the RMS distance and cosine of the angle between two sets of clusters in the physical space. It is assumed the clusters are in matrix/vector form.

    IMPORTANT!! It assumes the clusters are already ordered. (compares 1 with 1, ecc..)
    """

    et = []
    patcor = []

    for c1, c2 in zip(clus_1, clus_2):
        rms = LA.norm(c1-c2)
        et.append(rms)
        cosin = cosine(c1,c2)
        patcor.append(cosin)

    return et, patcor


def error2(pc1, pc2):
    """
    Compute root mean squared (RMS) error, phase error, and pattern correlation of two position vectors.
    """
    # RMS error
    RMS_error = LA.norm(pc1-pc2)

    # Compute the scalar product since it is needed to compute both the
    # phase error and the pattern correlation.
    dot = np.dot(pc1, pc2)

    # Compute the magnitude of each PC set.
    mag1 = LA.norm(pc1)
    mag2 = LA.norm(pc2)

    # Compute the phase error.
    error_phase = 2.0 * (mag1 * mag2 - dot)
    #ratio_EA_to_EP = (mag1-mag2)**2/error_phase

    # Compute the pattern correlation (cosine of the angle between the
    # two position vectors).
    pattern_correlation = dot / (mag1 * mag2)

    return RMS_error, error_phase, pattern_correlation


def match_pc_sets(pcset1, pcset2, debug=False):
    """Find the best possible match between two sets of PCs.

    Given two sets of PCs, finds the combination of PCs that minimizes
    the mean total squared error. It is intended that the input be sets
    of PCs representing cluster centroids. The results then correspond
    to the best unique match between the two sets of cluster centroids.

    The optimum combination will always have the first set of PCs in the
    input order, and re-arrange the second set of PCs. Therefore if you
    define the first PC set as those from model and the second set to be
    those from the reanalysis then the output ordering tells you the ERA
    cluster each model cluster best matches (in an optimal partition
    sense).

    Statistics are returned form each pattern in the optimum set so you
    can see what the individual values are.

    """
    if pcset1.shape != pcset2.shape:
        raise ValueError('both PC sets must have the same dimensionality')
    # Store the dimensionality of the PC sets.
    number_patterns, number_sets = pcset1.shape
    # Get generators for index pairs for each of the PC sets.
    iperm1, iperm2, nperms = pc_indices(number_patterns)
    # Create NumPy arrays to hold resulting statistics.
    errt = np.empty([nperms, number_patterns], dtype=pcset1.dtype)
    errp = np.empty([nperms, number_patterns], dtype=pcset1.dtype)
    pcor = np.empty([nperms, number_patterns], dtype=pcset1.dtype)
    perms = []
    # Loop over all permutations.
    for ip, (idx1, idx2) in enumerate(zip(iperm1, iperm2)):
        perms.append([idx1, idx2])
        # Compute the statistics for each pairing in this permutation.
        for iip, (i1, i2) in enumerate(zip(idx1, idx2)):
            et, ep, pc = error2(pcset1[i1], pcset2[i2])
            errt[ip, iip] = et
            errp[ip, iip] = ep
            pcor[ip, iip] = pc
    # Compute the mean squared error (mean of the squared error over each
    # pairing in a permutation), and find the permutation with the smallest
    # mean total squared error.
    mse = errt.mean(axis=1)
    jmin = mse.argmin()
    if debug:
        for i in xrange(pcor.shape[0]):
            print pcor[i], perms[i] + [invert(perms[i][1])], mse[i]
    # Get the statistics for the best match.
    return perms[jmin], errt[jmin], errp[jmin], pcor[jmin]

def invert(v1):
    l = len(v1)
    v2 = map(lambda i: np.where(np.array(v1)==i)[0][0], range(l))
    return v2

def match_clusters(input_var, input_file_1, input_file_2, output_file, npcs=None):
    #***CDAT
    # Open the input files and read the PCs.
    pcslicer = slice(0, npcs)
    try:
        fin1 = cdms2.open(input_file_1, 'r')
        pcset1 = fin1(input_var, pc=pcslicer)
    except cdms2.error.CDMSError:
        raise IOError('read error: %s' % input_file_1)
    try:
        fin2 = cdms2.open(input_file_2, 'r')
        pcset2 = fin2(input_var, pc=pcslicer)
    except cdms2.error.CDMSError:
        raise IOError('read error: %s' % input_file_2)
    # Determine the match statistics:
    # We should allow for multiple sets of PCs, for each cluster size.
    # Probably need to loop over them.
    clorder = np.ones([5, 6], dtype=np.float32) * np.nan
    error_total = np.ones([5, 6], dtype=np.float32) * np.nan
    error_phase = np.ones([5, 6], dtype=np.float32) * np.nan
    correlation = np.ones([5, 6], dtype=np.float32) * np.nan
    clorder = clorder.astype(np.float32)
    error_total = error_total.astype(np.float32)
    error_phase = error_phase.astype(np.float32)
    correlation = correlation.astype(np.float32)
    for k in range(2, 6+1):
        # Get the PC set for the current partition size.
        p1 = pcset1(partition_size=k, cluster_number=slice(0,k),
                squeeze=True).data
        p2 = pcset2(partition_size=k, cluster_number=slice(0,k),
                squeeze=True).data
        if k == 4:
            order, errt, errp, pcor = match_pc_sets(p1, p2, debug=False)
        else:
            order, errt, errp, pcor = match_pc_sets(p1, p2)
        # Define the ordering variable, this allows you to work out
        # which model cluster belongs to the corresponding ERA cluster.
        # If I find clorder(psize=4, clnum=3) = 1 I know that model cluster
        # 3 for k=4 matches ERA cluster 1. Also define statistics belonging
        # to each cluster centroid.
        for i, (o, et, ep, pc) in enumerate(zip(order[1], errt, errp, pcor)):
            clorder[k-2, i] = o
            error_total[k-2, i] = et
            error_phase[k-2, i] = ep
            correlation[k-2, i] = pc
    # Create cdms2 variables for each of the computed quantities.
    psize = cdms2.createAxis(range(2,7), id='partition_size')
    clnum = cdms2.createAxis(range(1,7), id='cluster_number')
    # Now construct the cdms2 variables you need.
    fill = np.float32(1.e20)
    #
    clorder[np.where(np.isnan(clorder))] = fill
    clorder = cdms2.createVariable(clorder, axes=[psize,clnum],
            id='clorder', fill_value=fill)
    clorder.name = ''
    clorder.long_name = 'cluster_correspondence'
    #
    error_total[np.where(np.isnan(error_total))] = fill
    error_total = cdms2.createVariable(error_total, axes=[psize,clnum],
            id='errort', fill_value=fill)
    error_total.name = 'errort'
    error_total.long_name = 'total_squared_error'
    #
    error_phase[np.where(np.isnan(error_phase))] = fill
    error_phase = cdms2.createVariable(error_phase, axes=[psize,clnum],
            id='errorp', fill_value=fill)
    error_phase.name = 'errorp'
    error_phase.long_name = 'phase_error'
    #
    correlation[np.where(np.isnan(correlation))] = fill
    correlation = cdms2.createVariable(correlation, axes=[psize,clnum],
            id='corr', fill_value=fill)
    correlation.name = 'corr'
    correlation.long_name = 'pattern_correlation_with_corresponding_cluster'
    # Open a file for appending.
    if output_file is not None:
        try:
            fout = cdms2.open(output_file, 'a')
        except:
            raise IOError('failed to open for writing: %s' % output_file)
        eclorder = fout['clorder']
        if eclorder is not None:
            eclorder.assignValue(clorder)
            fout.sync()
        else:
            fout.write(clorder)
        eerror_total = fout['errort']
        if eerror_total is not None:
            eerror_total.assignValue(error_total)
            fout.sync()
        else:
            fout.write(error_total)
        eerror_phase = fout['errorp']
        if eerror_phase is not None:
            eerror_phase.assignValue(error_phase)
            fout.sync()
        else:
            fout.write(error_phase)
        ecorrelation = fout['corr']
        if ecorrelation is not None:
            ecorrelation.assignValue(correlation)
            fout.sync()
        else:
            fout.write(correlation)
        fout.close()
#    # Print some stats to the screen.
#    print '                  Total Squared Error           ',
#    print '                      Phase Error               ',
#    print '                  Pattern Correlation           '
#    print 'k    CL1     CL2     CL3     CL4     CL5     CL6    ',
#    print 'CL1     CL2     CL3     CL4     CL5     CL6     ',
#    print 'CL1     CL2     CL3     CL4     CL5     CL6    '
#    for i in range(5):
#        print '%d   ' % (i+2),
#        for j in range(i+2):
#            print '%6.1f ' % error_total(partition_size=(i+2), cluster_number=(j+1)),
#        for j in range(i+2, 6):
#            print '   -   ',
#        for j in range(i+2):
#            print '%6.1f ' % error_phase(partition_size=(i+2), cluster_number=(j+1)),
#        for j in range(i+2, 6):
#            print '   -   ',
#        for j in range(i+2):
#            print '%5.2f  ' % correlation(partition_size=(i+2), cluster_number=(j+1)),
#        for j in range(i+2, 6):
#            print '   -   ',
#        print


def match_clusters_info(pcset1, pcset2, npcs=None):
    #***CDAT
    pcslicer = slice(0, npcs)
    # Determine the match statistics:
    # We should allow for multiple sets of PCs, for each cluster size.
    # Probably need to loop over them.
    clorder = np.ones([5, 6], dtype=np.float32) * np.nan
    error_total = np.ones([5, 6], dtype=np.float32) * np.nan
    error_phase = np.ones([5, 6], dtype=np.float32) * np.nan
    correlation = np.ones([5, 6], dtype=np.float32) * np.nan
    mse = np.ones([5], dtype=np.float32) * np.nan
    clorder = clorder.astype(np.float32)
    error_total = error_total.astype(np.float32)
    error_phase = error_phase.astype(np.float32)
    correlation = correlation.astype(np.float32)
    mse = mse.astype(np.float32)
    for k in range(2, 6+1):
        # Get the PC set for the current partition size.
        p1 = pcset1(partition_size=k, cluster_number=slice(0,k),
                squeeze=True).data
        p2 = pcset2(partition_size=k, cluster_number=slice(0,k),
                squeeze=True).data
        order, errt, errp, pcor = match_pc_sets(p2, p1)
        # Define the ordering variable, this allows you to work out
        # which model cluster belongs to the corresponding ERA cluster.
        # If I find clorder(psize=4, clnum=3) = 1 I know that model cluster
        # 3 for k=4 matches ERA cluster 1. Also define statistics belonging
        # to each cluster centroid.
        for i, (o, et, ep, pc) in enumerate(zip(order[1], errt, errp, pcor)):
            clorder[k-2, i] = o
            error_total[k-2, i] = et
            error_phase[k-2, i] = ep
            correlation[k-2, i] = pc
        mse[k-2] = (error_total[k-2, :k]).sum()
    # Create cdms2 variables for each of the computed quantities.
    psize = cdms2.createAxis(range(2,7), id='partition_size')
    clnum = cdms2.createAxis(range(1,7), id='cluster_number')
    # Now construct the cdms2 variables you need.
    fill = np.float32(1.e20)
    #
    clorder[np.where(np.isnan(clorder))] = fill
    clorder = cdms2.createVariable(clorder, axes=[psize,clnum],
            id='clorder', fill_value=fill)
    clorder.name = ''
    clorder.long_name = 'cluster_correspondence'
    #
    error_total[np.where(np.isnan(error_total))] = fill
    error_total = cdms2.createVariable(error_total, axes=[psize,clnum],
            id='errort', fill_value=fill)
    error_total.name = 'errort'
    error_total.long_name = 'total_squared_error'
    #
    error_phase[np.where(np.isnan(error_phase))] = fill
    error_phase = cdms2.createVariable(error_phase, axes=[psize,clnum],
            id='errorp', fill_value=fill)
    error_phase.name = 'errorp'
    error_phase.long_name = 'phase_error'
    #
    correlation[np.where(np.isnan(correlation))] = fill
    correlation = cdms2.createVariable(correlation, axes=[psize,clnum],
            id='corr', fill_value=fill)
    correlation.name = 'corr'
    correlation.long_name = 'pattern_correlation_with_corresponding_cluster'
    # Print some stats to the screen.
    print '                  Total Squared Error           ',
    print '                      Phase Error               ',
    print '                  Pattern Correlation           '
    print 'k    CL1     CL2     CL3     CL4     CL5     CL6    ',
    print 'CL1     CL2     CL3     CL4     CL5     CL6     ',
    print 'CL1     CL2     CL3     CL4     CL5     CL6    '
    for i in range(5):
        print '%d   ' % (i+2),
        for j in range(i+2):
            print '%6.1f ' % error_total(partition_size=(i+2), cluster_number=(j+1)),
        for j in range(i+2, 6):
            print '   -   ',
        for j in range(i+2):
            print '%6.1f ' % error_phase(partition_size=(i+2), cluster_number=(j+1)),
        for j in range(i+2, 6):
            print '   -   ',
        for j in range(i+2):
            print '%5.2f  ' % correlation(partition_size=(i+2), cluster_number=(j+1)),
        for j in range(i+2, 6):
            print '   -   ',
        print
    print 'MSE: k=2: %.1f  k=3: %.1f  k=4: %.1f k=5: %.1f k=6: %.1f' % tuple(mse)



if __name__ == '__main__':
    pcset1 = np.array([
                       [2, 2, 2, 2, 2],
                       [1, 1, 1, 1, 1],
                       [3, 3, 3, 3, 3],
                      ], dtype=np.float32)
    pcset2 = np.array([
                       [4, 4, 4, 4, 4],
                       [2, 2, 2, 2, 2],
                       [1, 1, 1, 1, 1],
                      ], dtype=np.float32)
    p, et, ep, pc = match_pc_sets(pcset1, pcset2)
    print et
    print ep
    print pc
    print p
