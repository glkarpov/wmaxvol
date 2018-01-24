#! /bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sympy import *

### Main fucns

def SymbVars(l, name='x'):
    return symbols(StrVars(l, name=name, last_comma=True))

def StrVars(l, name='x', delim=',', last_comma=False, add_on = ''):
    lst = ',' if last_comma else '' # last ",", gives tuple anywhere, including l==1
    return delim.join([name + str(i) + add_on for i in xrange(l)]) + lst

def export_f_txt(func, l):
    """
    Export sympy object as string
    INPUT
        func -- sympy object
        l    -- number of vars
    """
    # vars_f = symbols(' '.join(['x' + str(i) for i in xrange(l)]))
    vars_f = SymbVars(l)
    return str(simplify(func(*vars_f)))

def export_f_Math(func, l, func_name='F'):
    """
    Export sympy object as Mathematica function
    """
    ans = export_f_txt(func, l)
    # vars_f = ', '.join(['x' + str(i) + '_' for i in xrange(l)])
    vars_f = StrVars(l, delim=', ', add_on = '_')
    return "{}[{}]:={}".format(func_name, vars_f, ans)

def symb_to_func_old(func, l, is_func=True):
    """
    INPUT
        func -- function
        l    -- dimension
        is_func  -- whether func is usual func (True) or sympy object (False)
    RETURNS 
        lambda - func from synpy obj
    """
    if is_func:
        ans = export_f_txt(func, l)
    else:
        ans = str(func)
    #replaces
    for i in ['exp','cos', 'sin']:
        ans = ans.replace(i + '(', 'np.' + i +'(')

    # vars_f = ', '.join(['x' + str(i) for i in xrange(l)])
    vars_f = StrVars(l, delim=', ')
    return eval("lambda  " + vars_f + " :  " + ans  ), ans

def symb_to_func(func, l, is_func=True):
    """
    INPUT
        func -- function
        l    -- dimension
        is_func -- shows, whether func is lambda func of expression
    RETURNS 
        lambda - func from sympy obj
    """
    # vars_f = symbols(   ', '.join([ 'x' + str(i) for i in xrange(l) ])  )
    vars_f = SymbVars(l)
    if is_func:
        return utilities.lambdify(vars_f, func(*vars_f), 'numpy'), export_f_txt(func, l)
    else:
        return utilities.lambdify(vars_f, func, 'numpy'), str(func)

    

def find_brack(s, i):
    # find corresponding brackets (square and circle)
    ret = -1
    fnd = None
    br_cur = s[i]
    if br_cur == '(':
        fnd = ')'
    if br_cur == '[':
        fnd = ']'
    if fnd is None:
        return None

    # print s, i, s[i]
    br_count = 0
    while i < len(s):
        if s[i] == br_cur:
            br_count += 1
        if s[i] == fnd:
            br_count -= 1
            if br_count == 0:
                return i
                break
        i += 1

    return -1

def To_math(s):
    """
    One of the main function
    Add other functions below
    """
    srch_str = ['np.exp', 'exp']
    repl_str = ['Exp', 'Exp']

    for i_cur_str, cur_str in enumerate(srch_str):
        i = s.find(cur_str)
        i_old = i
        while i > -1:
            i = i + len(cur_str)
            while   s[i] != '('   and   i < len(s):
                i += 1
            if i < len(s):
                idx = find_brack(s, i)
                if idx is not None and idx > -1:
                    s = s[:i_old] + repl_str[i_cur_str]  + '[' + s[i+1:idx] + ']' + s[idx+1:]
                else:
                    print 'Bad String'
                    return s

            i = s.find(cur_str)

    return s

def FindDiff(f, d, i=1, ret_symb=False):
    """
        f -- function
        d -- dimension
        i -- diff the func w.r.t. ith arg, i<=d
        ret_symb -- whether to return symbolic representation along with func
    """
    vars_f = SymbVars(d)
    fd = diff(f(*vars_f), vars_f[i-1])
    fout, _ = symb_to_func(fd, d, False)
    if ret_symb:
        return fout, fd
    else:
        return fout


if __name__ == '__main__':
    print 'Test run'
    def f(x, y):
        return x**6+2*y

    ret = export_f_txt(f, 2)
    print "Result of export_f_txt", type(ret), ret

    from numpy.polynomial import Chebyshev as T

    def f2(x, y):
        return T.basis(6)(x) + y

    ret = export_f_txt(f2, 2)
    print "Chebyshev poly", ret


    ret = export_f_Math(f2, 2)
    print "export_f_Math", ret


    print "Differentiation"
    def f3(x, y):
        return T.basis(3)(x) + y*y


    ret = export_f_txt(f3, 2)
    print "Chebyshev poly", ret
    fdiff, fdiff_symb = FindDiff(f3, 2, 1, True)
    fdiff2, fdiff_symb2 = FindDiff(f3, 2, 2, True)
    print fdiff_symb, "\n", fdiff_symb2

    import matplotlib.pyplot as plt
    fig = plt.figure()
    xp = np.linspace(-1,1,1000)
    yp = f3(xp, 0)
    yp_diff = fdiff(xp, 0)

    plt.plot(xp, yp, xp, yp_diff)
    # plt.show()


    # Math functions form numpy ans sympy
    print "-"*70

    def f_sin(x):
        return sin(x) # Actually, its sympy.sin


    f_sin_l, _ = symb_to_func(f_sin, 1) # Make real (numpy) func from sympy objects

    f_diff, f_diff_symb = FindDiff(f_sin, 1, 1, True) # U can pass either f_sin or f_sin_l here
    print "Diff of func with sin:", f_diff_symb

    fig = plt.figure()
    xp = np.linspace(-np.pi, np.pi, 1000)
    yp = f_sin_l(xp) # U can use only f_sin_l here, not f_sin!!!
    yp_diff = f_diff(xp)

    plt.plot(xp, yp, xp, yp_diff)
    plt.show()

