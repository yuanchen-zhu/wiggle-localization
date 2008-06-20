#include <Python.h> // needed for all extensions

#include "structmember.h" // for custom type
#include "numpy/arrayobject.h" // for numpy array1


#include "arpack++/arlsmat.h"
#include "arpack++/arlssym.h"

using namespace std;

static bool arrayCheck(PyArrayObject *o, int requiredRank, int requiredType)
{
    return o->nd == requiredRank && o->descr->type_num == requiredType;
}


static PyObject *sparse_eig(PyObject* *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = { "n", "nzval", "irow", "pcol", "lu", "nev", "which", "eigval", "eigvec", "maxit", NULL};
    PyArrayObject *nzval = NULL, *irow = NULL, *pcol = NULL, *eigval = NULL, *eigvec = NULL;
    int n, nev, maxit;
    char *which, *lu;

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "iO!O!O!sisO!O!i", kwlist,
                                      &n,
                                      &PyArray_Type, &nzval,
                                      &PyArray_Type, &irow,
                                      &PyArray_Type, &pcol,
                                      &lu,
                                      &nev,
                                      &which,
                                      &PyArray_Type, &eigval,
                                      &PyArray_Type, &eigvec,
                                      &maxit))
        return NULL;

    if (!arrayCheck(nzval, 1, NPY_DOUBLE) ||
        nzval->strides[0] != sizeof(npy_double)) {
        PyErr_SetString(PyExc_ValueError, "argument `nzval' must be a continguous 1D NumPy array of doubles");
        return NULL;
    }

    if (!arrayCheck(irow, 1, NPY_INT) ||
        irow->strides[0] != sizeof(int)) {
        PyErr_SetString(PyExc_ValueError, "argument `irow' must be a continguous 1D NumPy array of integers");
        return NULL;
    }

    if (!arrayCheck(pcol, 1, NPY_INT) ||
        irow->strides[0] != sizeof(int)) {
        PyErr_SetString(PyExc_ValueError, "argument `pcol' must be a contiguous 1D NumPy array of integers");
        return NULL;
    }

    if (!arrayCheck(eigval, 1, NPY_DOUBLE) ||
        eigval->dimensions[0] < nev) {
        PyErr_SetString(PyExc_ValueError, "argument `eigval' must be a 1D NumPy array of doubles of size at least `nev'");
        return NULL;
    }
        
    if (!arrayCheck(eigvec, 2, NPY_DOUBLE) ||
        eigvec->dimensions[0] < n ||
        eigvec->dimensions[1] < nev) {
        PyErr_SetString(PyExc_ValueError, "argument `eigvec' must be a 2D NumPy array of doubles of size at least `n` by `nev' ");
        return NULL;
    }

    typedef ARluSymMatrix<double> Matrix;
    typedef ARluSymStdEig<double> EigProb;
    Matrix A(n, nzval->dimensions[0], (double*)nzval->data, (int*)irow->data, (int*)pcol->data, lu[0]);
    EigProb prob(nev, A, which);
    prob.ChangeMaxit(maxit);
    prob.FindEigenvectors();
    int d = prob.GetN();
    for (int i = 0, n = prob.ConvergedEigenvalues(); i < n; ++i) {
        *(double*)(eigval->data + i * eigval->strides[0]) = prob.Eigenvalue(i);
        char *p = eigvec->data + i * eigvec->strides[1];
        for (int j = 0; j < d; ++j, p += eigvec->strides[0])
            *(double*)p = prob.Eigenvector(i, j);
    }
    Py_INCREF(Py_None);
    return Py_None;
}

static char *speig_doc =
"speig(n, nzval, irow, pcol, lu, nev, which, eigval, eigvec, maxit):\n\
sparse eigen value and vector calculation.";

static PyMethodDef sparselin_methods[] = {
    {"speig",  (PyCFunction)sparse_eig,    METH_VARARGS|METH_KEYWORDS,    speig_doc},
    {NULL,      NULL,                       0,              NULL} // sentinel 
};
    
extern "C" {

    void initsparselin()
    {
        // Create the module and add the functions
        PyObject *m = Py_InitModule("sparselin", sparselin_methods);
        
        import_array();

        // Check for errors
        if (m == NULL || PyErr_Occurred())
            Py_FatalError("Can't initialize module sparselin");
    }
}


    



