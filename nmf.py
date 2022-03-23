# Authors: Christian Thurau
# License: BSD 3 Clause
"""
PyMF Non-negative Matrix Factorization.
    NMF: Class for Non-negative Matrix Factorization
[1] Lee, D. D. and Seung, H. S. (1999), Learning the Parts of Objects by Non-negative
Matrix Factorization, Nature 401(6755), 788-799.
"""
import numpy as np
import logging
import logging.config
import scipy.sparse
import scipy.optimize
from cvxopt import solvers, base

# Authors: Christian Thurau
# License: BSD 3 Clause
"""
PyMF base class used in (almost) all matrix factorization methods
"""
import numpy as np
import logging
import logging.config
import scipy.sparse
from numpy.linalg import eigh
from scipy.special import factorial

# Authors: Christian Thurau
# License: BSD 3 Clause
"""  
PyMF Singular Value Decomposition.
    SVD : Class for Singular Value Decomposition
    pinv() : Compute the pseudoinverse of a Matrix
     
"""
from numpy.linalg import eigh
import time
import scipy.sparse
import numpy as np

try:
    import scipy.sparse.linalg.eigen.arpack as linalg
except (ImportError, AttributeError):
    import scipy.sparse.linalg as linalg


_EPS = np.finfo(float).eps

def eighk(M, k=0):
    """ Returns ordered eigenvectors of a squared matrix. Too low eigenvectors
    are ignored. Optionally only the first k vectors/values are returned.
    Arguments
    ---------
    M - squared matrix
    k - (default 0): number of eigenvectors/values to return
    Returns
    -------
    w : [:k] eigenvalues 
    v : [:k] eigenvectors
    """
    values, vectors = eigh(M)            
              
    # get rid of too low eigenvalues
    s = np.where(values > _EPS)[0]
    vectors = vectors[:, s] 
    values = values[s]                            
             
    # sort eigenvectors according to largest value
    idx = np.argsort(values)[::-1]
    values = values[idx]
    vectors = vectors[:,idx]

    # select only the top k eigenvectors
    if k > 0:
        values = values[:k]
        vectors = vectors[:,:k]

    return values, vectors


def cmdet(d):
    """ Returns the Volume of a simplex computed via the Cayley-Menger
    determinant.
    Arguments
    ---------
    d - euclidean distance matrix (shouldn't be squared)
    Returns
    -------
    V - volume of the simplex given by d
    """
    D = np.ones((d.shape[0]+1,d.shape[0]+1))
    D[0,0] = 0.0
    D[1:,1:] = d**2
    j = np.float32(D.shape[0]-2)
    f1 = (-1.0)**(j+1) / ( (2**j) * ((factorial(j))**2))
    cmd = f1 * np.linalg.det(D)

    # sometimes, for very small values, "cmd" might be negative, thus we take
    # the absolute value
    return np.sqrt(np.abs(cmd))


def simplex(d):
    """ Computed the volume of a simplex S given by a coordinate matrix D.
    Arguments
    ---------
    d - coordinate matrix (k x n, n samples in k dimensions)
    Returns
    -------
    V - volume of the Simplex spanned by d
    """
    # compute the simplex volume using coordinates
    D = np.ones((d.shape[0]+1, d.shape[1]))
    D[1:,:] = d
    V = np.abs(np.linalg.det(D)) / factorial(d.shape[1] - 1)
    return V


class PyMFBase():
    """
    PyMF Base Class. Does nothing useful apart from poviding some basic methods.
    """
    # some small value
   
    _EPS = _EPS
    
    def __init__(self, data, num_bases=4, **kwargs):
        """
        """
        
        def setup_logging():
            # create logger       
            self._logger = logging.getLogger("pymf")
       
            # add ch to logger
            if len(self._logger.handlers) < 1:
                # create console handler and set level to debug
                ch = logging.StreamHandler()
                ch.setLevel(logging.DEBUG)
                # create formatter
                formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        
                # add formatter to ch
                ch.setFormatter(formatter)

                self._logger.addHandler(ch)

        setup_logging()
        
        # set variables
        self.data = data       
        self._num_bases = num_bases             
      
        # initialize H and W to random values
        self._data_dimension, self._num_samples = self.data.shape
        

    def residual(self):
        """ Returns the residual in % of the total amount of data
        Returns
        -------
        residual : float
        """
        res = np.sum(np.abs(self.data - np.dot(self.W, self.H)))
        total = 100.0*res/np.sum(np.abs(self.data))
        return total
        
    def frobenius_norm(self):
        """ Frobenius norm (||data - WH||) of a data matrix and a low rank
        approximation given by WH. Minimizing the Fnorm ist the most common
        optimization criterion for matrix factorization methods.
        Returns:
        -------
        frobenius norm: F = ||data - WH||
        """
        # check if W and H exist
        if hasattr(self,'H') and hasattr(self,'W'):
            if scipy.sparse.issparse(self.data):
                tmp = self.data[:,:] - (self.W * self.H)
                tmp = tmp.multiply(tmp).sum()
                err = np.sqrt(tmp)
            else:
                err = np.sqrt( np.sum((self.data[:,:] - np.dot(self.W, self.H))**2 ))            
        else:
            err = None

        return err
        
    def _init_w(self):
        """ Initalize W to random values [0,1].
        """
        # add a small value, otherwise nmf and related methods get into trouble as 
        # they have difficulties recovering from zero.
        self.W = np.random.random((self._data_dimension, self._num_bases)) + 10**-4
        
    def _init_h(self):
        """ Initalize H to random values [0,1].
        """
        self.H = np.random.random((self._num_bases, self._num_samples)) + 10**-4
        
    def _update_h(self):
        """ Overwrite for updating H.
        """
        pass

    def _update_w(self):
        """ Overwrite for updating W.
        """
        pass

    def _converged(self, i):
        """ 
        If the optimization of the approximation is below the machine precision,
        return True.
        Parameters
        ----------
            i   : index of the update step
        Returns
        -------
            converged : boolean 
        """
        derr = np.abs(self.ferr[i] - self.ferr[i-1])/self._num_samples
        if derr < self._EPS:
            return True
        else:
            return False

    def factorize(self, niter=100, show_progress=False, 
                  compute_w=True, compute_h=True, compute_err=True):
        """ Factorize s.t. WH = data
        
        Parameters
        ----------
        niter : int
                number of iterations.
        show_progress : bool
                print some extra information to stdout.
        compute_h : bool
                iteratively update values for H.
        compute_w : bool
                iteratively update values for W.
        compute_err : bool
                compute Frobenius norm |data-WH| after each update and store
                it to .ferr[k].
        
        Updated Values
        --------------
        .W : updated values for W.
        .H : updated values for H.
        .ferr : Frobenius norm |data-WH| for each iteration.
        """
        
        if show_progress:
            self._logger.setLevel(logging.INFO)
        else:
            self._logger.setLevel(logging.ERROR)        
        
        # create W and H if they don't already exist
        # -> any custom initialization to W,H should be done before
        if not hasattr(self,'W') and compute_w:
            self._init_w()
               
        if not hasattr(self,'H') and compute_h:
            self._init_h()                   
        
        # Computation of the error can take quite long for large matrices,
        # thus we make it optional.
        if compute_err:
            self.ferr = np.zeros(niter)
             
        for i in range(niter):
            if compute_w:
                self._update_w()

            if compute_h:
                self._update_h()                                        
         
            if compute_err:                 
                self.ferr[i] = self.frobenius_norm()                
                self._logger.info('FN: %s (%s/%s)'  %(self.ferr[i], i+1, niter))
            else:                
                self._logger.info('Iteration: (%s/%s)'  %(i+1, niter))
           

            # check if the err is not changing anymore
            if i > 1 and compute_err:
                if self._converged(i):
                    # adjust the error measure
                    self.ferr = self.ferr[:i]
                    break


class PyMFBase3():    
    """      
    PyMFBase3(data, show_progress=False)
    
    Base class for factorizing a data matrix into three matrices s.t. 
    F = | data - USV| is minimal (e.g. SVD, CUR, ..)
    
    Parameters
    ----------
    data : array_like [data_dimension x num_samples]
        the input data
    
    Attributes
    ----------
        U,S,V : submatrices s.t. data = USV                
    
    """
    _EPS = _EPS

    
    def __init__(self, data, k=-1, rrank=0, crank=0):
        """
        """
        self.data = data
        (self._rows, self._cols) = self.data.shape

        self._rrank = self._rows
        if rrank > 0:
            self._rrank = rrank
            
        self._crank = self._cols
        if crank > 0:            
            self._crank = crank
        
        self._k = k
    
    def frobenius_norm(self):
        """ Frobenius norm (||data - USV||) for a data matrix and a low rank
        approximation given by SVH using rank k for U and V
        
        Returns:
            frobenius norm: F = ||data - USV||
        """    
        if scipy.sparse.issparse(self.data):
            err = self.data - (self.U*self.S*self.V)
            err = err.multiply(err)
            err = np.sqrt(err.sum())
        else:                
            err = self.data[:,:] - np.dot(np.dot(self.U, self.S), self.V)
            err = np.sqrt(np.sum(err**2))
                            
        return err
        
    
    def factorize(self):    
        pass

def _test():
    import doctest
    doctest.testmod()
    
def pinv(A, k=-1, eps= np.finfo(float).eps):    
    # Compute Pseudoinverse of a matrix   
    svd_mdl =  SVD(A, k=k)
    svd_mdl.factorize()
    
    S = svd_mdl.S
    Sdiag = S.diagonal()
    Sdiag = np.where(Sdiag>eps, 1.0/Sdiag, 0.0)
    
    for i in range(S.shape[0]):
        S[i,i] = Sdiag[i]

    if scipy.sparse.issparse(A):            
        A_p = svd_mdl.V.transpose() * (S * svd_mdl.U.transpose())
    else:    
        A_p = np.dot(svd_mdl.V.T, np.core.multiply(np.diag(S)[:,np.newaxis], svd_mdl.U.T))

    return A_p


class SVD(PyMFBase3):    
    """      
    SVD(data, show_progress=False)
    
    
    Singular Value Decomposition. Factorize a data matrix into three matrices s.t.
    F = | data - USV| is minimal. U and V correspond to eigenvectors of the matrices
    data*data.T and data.T*data.
    
    Parameters
    ----------
    data : array_like [data_dimension x num_samples]
        the input data
    
    Attributes
    ----------
        U,S,V : submatrices s.t. data = USV                
    
    Example
    -------
    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> svd_mdl = SVD(data)    
    >>> svd_mdl.factorize()
    """

    def _compute_S(self, values):
        """
        """
        self.S = np.diag(np.sqrt(values))
        
        # and the inverse of it
        S_inv = np.diag(np.sqrt(values)**-1.0)
        return S_inv

   
    def factorize(self):    

        def _right_svd():            
            AA = np.dot(self.data[:,:], self.data[:,:].T)
                   # argsort sorts in ascending order -> access is backwards
            values, self.U = eighk(AA, k=self._k)

            # compute S
            self.S = np.diag(np.sqrt(values))
            
            # and the inverse of it
            S_inv = self._compute_S(values)
                    
            # compute V from it
            self.V = np.dot(S_inv, np.dot(self.U[:,:].T, self.data[:,:]))    
            
        
        def _left_svd():
            AA = np.dot(self.data[:,:].T, self.data[:,:])
            
            values, Vtmp = eighk(AA, k=self._k)
            self.V = Vtmp.T 

            # and the inverse of it
            S_inv = self._compute_S(values)

            self.U = np.dot(np.dot(self.data[:,:], self.V.T), S_inv)                
    
        def _sparse_right_svd():
            ## for some reasons arpack does not allow computation of rank(A) eigenvectors (??)    #
            AA = self.data*self.data.transpose()
            
            if self.data.shape[0] > 1:                    
                # only compute a few eigenvectors ...
                if self._k > 0 and self._k < self.data.shape[0]-1:
                    k = self._k
                else:
                    k = self.data.shape[0]-1
                values, u_vectors = linalg.eigsh(AA,k=k)
            else:                
                values, u_vectors = eigh(AA.todense())
            
            # get rid of negative/too low eigenvalues   
            s = np.where(values > self._EPS)[0]
            u_vectors = u_vectors[:, s] 
            values = values[s]
            
            # sort eigenvectors according to largest value
            # argsort sorts in ascending order -> access is backwards
            idx = np.argsort(values)[::-1]
            values = values[idx]                        
            
            self.U = scipy.sparse.csc_matrix(u_vectors[:,idx])
                    
            # compute S
            tmp_val = np.sqrt(values)            
            l = len(idx)
            self.S = scipy.sparse.spdiags(tmp_val, 0, l, l,format='csc') 
            
            # and the inverse of it            
            S_inv = scipy.sparse.spdiags(1.0/tmp_val, 0, l, l,format='csc')
            
            # compute V from it
            self.V = self.U.transpose() * self.data
            self.V = S_inv * self.V
    
        def _sparse_left_svd():        
            # for some reasons arpack does not allow computation of rank(A) eigenvectors (??)
            AA = self.data.transpose()*self.data
    
            if self.data.shape[1] > 1:                
                # do not compute full rank if desired
                if self._k > 0 and self._k < AA.shape[1]-1:
                    k = self._k
                else:
                    k = self.data.shape[1]-1
                
                values, v_vectors = linalg.eigsh(AA,k=k)                    
            else:                
                values, v_vectors = eigh(AA.todense())    
           
            # get rid of negative/too low eigenvalues   
            s = np.where(values > self._EPS)[0]
            v_vectors = v_vectors[:, s] 
            values = values[s]
            
            # sort eigenvectors according to largest value
            idx = np.argsort(values)[::-1]                  
            values = values[idx]
            
            # argsort sorts in ascending order -> access is backwards            
            self.V = scipy.sparse.csc_matrix(v_vectors[:,idx])      
            
            # compute S
            tmp_val = np.sqrt(values)            
            l = len(idx)      
            self.S = scipy.sparse.spdiags(tmp_val, 0, l, l,format='csc') 
            
            # and the inverse of it                                         
            S_inv = scipy.sparse.spdiags(1.0/tmp_val, 0, l, l,format='csc')
            
            self.U = self.data * self.V * S_inv        
            self.V = self.V.transpose()           
        
        if self._rows >= self._cols:
            if scipy.sparse.issparse(self.data):                
                _sparse_left_svd()
            else:            
                _left_svd()
        else:
            if scipy.sparse.issparse(self.data):
                _sparse_right_svd()
            else:            
                _right_svd()

def _test():
    import doctest
    doctest.testmod()


class NMF(PyMFBase):
    """
    NMF(data, num_bases=4)
    Non-negative Matrix Factorization. Factorize a data matrix into two matrices
    s.t. F = | data - W*H | = | is minimal. H, and W are restricted to non-negative
    data. Uses the classicial multiplicative update rule.
    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)        
    Attributes
    ----------
    W : "data_dimension x num_bases" matrix of basis vectors
    H : "num bases x num_samples" matrix of coefficients
    ferr : frobenius norm (after calling .factorize()) 
    Example
    -------
    Applying NMF to some rather stupid data set:
    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> nmf_mdl = NMF(data, num_bases=2, niter=10)
    >>> nmf_mdl.factorize()
    The basis vectors are now stored in nmf_mdl.W, the coefficients in nmf_mdl.H.
    To compute coefficients for an existing set of basis vectors simply    copy W
    to nmf_mdl.W, and set compute_w to False:
    >>> data = np.array([[1.5], [1.2]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> nmf_mdl = NMF(data, num_bases=2)
    >>> nmf_mdl.W = W
    >>> nmf_mdl.factorize(niter=20, compute_w=False)
    The result is a set of coefficients nmf_mdl.H, s.t. data = W * nmf_mdl.H.
    """
       
    def _update_h(self):
        # pre init H1, and H2 (necessary for storing matrices on disk)
        H2 = np.dot(np.dot(self.W.T, self.W), self.H) + 10**-9
        self.H *= np.dot(self.W.T, self.data[:,:])
        self.H /= H2

    def _update_w(self):
        # pre init W1, and W2 (necessary for storing matrices on disk)
        W2 = np.dot(np.dot(self.W, self.H), self.H.T) + 10**-9
        self.W *= np.dot(self.data[:,:], self.H.T)
        self.W /= W2
        self.W /= np.sqrt(np.sum(self.W**2.0, axis=0))


class RNMF(PyMFBase):
    """
    RNMF(data, num_bases=4)
    Non-negative Matrix Factorization. Factorize a data matrix into two matrices
    s.t. F = | data - W*H | = | is minimal. H, and W are restricted to non-negative
    values. Uses the classicial multiplicative update rule.
    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)        
    Attributes
    ----------
    W : "data_dimension x num_bases" matrix of basis vectors
    H : "num bases x num_samples" matrix of coefficients
    ferr : frobenius norm (after calling .factorize()) 
    Example
    -------
    Applying NMF to some rather stupid data set:
    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> nmf_mdl = RNMF(data, num_bases=2)
    >>> nmf_mdl.factorize()
    The basis vectors are now stored in nmf_mdl.W, the coefficients in nmf_mdl.H.
    To compute coefficients for an existing set of basis vectors simply    copy W
    to nmf_mdl.W, and set compute_w to False:
    >>> data = np.array([[1.5], [1.2]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> nmf_mdl = RNMF(data, num_bases=2)
    >>> nmf_mdl.W = W
    >>> nmf_mdl.factorize(niter=20, compute_w=False)
    The result is a set of coefficients nmf_mdl.H, s.t. data = W * nmf_mdl.H.
    """
    
    def __init__(self, data, num_bases=4, lamb=2.0):
        # call inherited method
        PyMFBase.__init__(self, data, num_bases=num_bases)
        self._lamb = lamb
    
    def soft_thresholding(self, X, lamb):       
        X = np.where(np.abs(X) <= lamb, 0.0, X)
        X = np.where(X > lamb, X - lamb, X)
        X = np.where(X < -1.0*lamb, X + lamb, X)
        return X
        
    def _init_h(self):
        self.H = np.random.random((self._num_bases, self._num_samples))
        self.H[:,:] = 1.0

        # normalized bases
        Wnorm = np.sqrt(np.sum(self.W**2.0, axis=0))
        self.W /= Wnorm
        
        for i in range(self.H.shape[0]):
            self.H[i,:] *= Wnorm[i]
            
        self._update_s()
        
    def _update_s(self):                
        self.S = self.data - np.dot(self.W, self.H)
        self.S = self.soft_thresholding(self.S, self._lamb)
    
    def _update_h(self):
        # pre init H1, and H2 (necessary for storing matrices on disk)
        H1 = np.dot(self.W.T, self.S - self.data)
        H1 = np.abs(H1) - H1
        H1 /= (2.0* np.dot(self.W.T, np.dot(self.W, self.H)))        
        self.H *= H1
  
        # adapt S
        self._update_s()
  
    def _update_w(self):
        # pre init W1, and W2 (necessary for storing matrices on disk)
        W1 = np.dot(self.S - self.data, self.H.T)
        #W1 = np.dot(self.data - self.S, self.H.T)       
        W1 = np.abs(W1) - W1
        W1 /= (2.0 * (np.dot(self.W, np.dot(self.H, self.H.T))))
        self.W *= W1           



class NMFALS(PyMFBase):
    """      
    NMFALS(data, num_bases=4)
    
    
    Non-negative Matrix Factorization. Factorize a data matrix into two matrices 
    s.t. F = | data - W*H | = | is minimal. H, and W are restricted to non-negative
    data. Uses the an alternating least squares procedure (quite slow for larger
    data sets) and cvxopt, similar to aa.
    
    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)    
    
    Attributes
    ----------
    W : "data_dimension x num_bases" matrix of basis vectors
    H : "num bases x num_samples" matrix of coefficients
    ferr : frobenius norm (after calling .factorize())     
    
    Example
    -------
    Applying NMF to some rather stupid data set:
    
    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> nmf_mdl = NMFALS(data, num_bases=2)
    >>> nmf_mdl.factorize(niter=10)
    
    The basis vectors are now stored in nmf_mdl.W, the coefficients in nmf_mdl.H. 
    To compute coefficients for an existing set of basis vectors simply    copy W 
    to nmf_mdl.W, and set compute_w to False:
    
    >>> data = np.array([[1.5], [1.2]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> nmf_mdl = NMFALS(data, num_bases=2)
    >>> nmf_mdl.W = W
    >>> nmf_mdl.factorize(niter=1, compute_w=False)
    
    The result is a set of coefficients nmf_mdl.H, s.t. data = W * nmf_mdl.H.
    """
 
    def _update_h(self):
        def updatesingleH(i):
            # optimize alpha using qp solver from cvxopt
            FA = base.matrix(np.float64(np.dot(-self.W.T, self.data[:,i])))
            al = solvers.qp(HA, FA, INQa, INQb)
            self.H[:,i] = np.array(al['x']).reshape((1,-1))
                                                                
        # float64 required for cvxopt
        HA = base.matrix(np.float64(np.dot(self.W.T, self.W)))            
        INQa = base.matrix(-np.eye(self._num_bases))
        INQb = base.matrix(0.0, (self._num_bases,1))            
    
        map(updatesingleH, range(self._num_samples))                        
            
                
    def _update_w(self):
        def updatesingleW(i):
        # optimize alpha using qp solver from cvxopt
            FA = base.matrix(np.float64(np.dot(-self.H, self.data[i,:].T)))
            al = solvers.qp(HA, FA, INQa, INQb)                
            self.W[i,:] = np.array(al['x']).reshape((1,-1))            
                                
        # float64 required for cvxopt
        HA = base.matrix(np.float64(np.dot(self.H, self.H.T)))                    
        INQa = base.matrix(-np.eye(self._num_bases))
        INQb = base.matrix(0.0, (self._num_bases,1))            

        map(updatesingleW, range(self._data_dimension))

        self.W = self.W/np.sum(self.W, axis=1)


class NMFNNLS(PyMFBase):
    """      
    NMFNNLS(data, num_bases=4)
    
    
    Non-negative Matrix Factorization. Factorize a data matrix into two matrices 
    s.t. F = | data - W*H | = | is minimal. H, and W are restricted to non-negative
    data. Uses the Lawsons and Hanson's algorithm for non negative constrained
    least squares (-> also see scipy.optimize.nnls)
    
    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)    
    
    Attributes
    ----------
    W : "data_dimension x num_bases" matrix of basis vectors
    H : "num bases x num_samples" matrix of coefficients
    ferr : frobenius norm (after calling .factorize())     
    
    Example
    -------
    Applying NMF to some rather stupid data set:
    
    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> nmf_mdl = NMFNNLS(data, num_bases=2)
    >>> nmf_mdl.factorize(niter=10)
    
    The basis vectors are now stored in nmf_mdl.W, the coefficients in nmf_mdl.H. 
    To compute coefficients for an existing set of basis vectors simply copy W 
    to nmf_mdl.W, and set compute_w to False:
    
    >>> data = np.array([[1.5], [1.2]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> nmf_mdl = NMFNNLS(data, num_bases=2)
    >>> nmf_mdl.W = W
    >>> nmf_mdl.factorize(niter=1, compute_w=False)
    
    The result is a set of coefficients nmf_mdl.H, s.t. data = W * nmf_mdl.H.
    """

    def _update_h(self):
        def updatesingleH(i):        
            self.H[:,i] = scipy.optimize.nnls(self.W, self.data[:,i])[0]
                                                                            
        map(updatesingleH, range(self._num_samples))                        
            
                
    def _update_w(self):
        def updatesingleW(i):            
            self.W[i,:] = scipy.optimize.nnls(self.H.T, self.data[i,:].T)[0]

        map(updatesingleW, range(self._data_dimension))


def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()