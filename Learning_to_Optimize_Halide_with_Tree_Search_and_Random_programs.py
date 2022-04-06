import numpy as np
from pytest import param
import math
from util import *

def raw_major_order(params):
    # 3 navigating the halides scheduling space
    # import pdb;pdb.set_trace()
    M,N,h,f = params
    g = np.power(h, 1.8)
    for m in range(M):
        for n in range(N):
            if n+1 >= N:
                N_plus_1 = 0
            else:
                N_plus_1 = n+1
            # print(m,n-1,"+",m,N_plus_1)
            f[m,n] = g[m,n-1] + g[m,N_plus_1]
    # print(f)
    return f

def SIMD(f, g, params):
    # single instruction multiply data
    VM = 1
    VN = 8
    tiled,n_o,PN,n_i_o,N,m_o,PM,m_i_o = params
    for m_i_i in range(VM):
        for n_i_i in range(VN):
            if (n_o*PN + (n_i_o*VN + n_i_i))+1 >= N:
                N_plus_1 = 0
            else:
                N_plus_1 = n_o*PN + (n_i_o*VN + n_i_i) + 1
            if tiled:
                f[(m_o*PM + (m_i_o*VM + m_i_i)),(n_o*PN + (n_i_o*VN + n_i_i))] = g[(m_i_o*VM + m_i_i), (n_i_o*VN + n_i_i)-1] + g[(m_i_o*VM + m_i_i),N_plus_1]
            else:
                f[(m_o*PM + (m_i_o*VM + m_i_i)),(n_o*PN + (n_i_o*VN + n_i_i))] = g[(m_o*PM + (m_i_o*VM + m_i_i)),(n_o*PN + (n_i_o*VN + n_i_i))-1] + g[(m_o*PM + (m_i_o*VM + m_i_i)),N_plus_1]
    return

def parallel_vectorize_tile(params):
    # import pdb;pdb.set_trace()
    M,N,h,f = params
    PM = 4
    PN = 16
    VM = 1
    VN = 8
    g = np.power(h, 1.8)
    for m_o in range(M//PM):
        for n_o in range(N//PN):
            for m_i_o in range(PM//VM):
                for n_i_o in range(PN//VN):
                    params = 0,n_o,PN,n_i_o,N,m_o,PM,m_i_o
                    SIMD(f, g, params)
                    # for m_i_i in range(VM):
                    #     for n_i_i in range(VN):
                    #         if (n_o*PN + (n_i_o*VN + n_i_i))+1 >= N:
                    #             N_plus_1 = 0
                    #         else:
                    #             N_plus_1 = n_o*PN + (n_i_o*VN + n_i_i) + 1
                    #         f[(m_o*PM + (m_i_o*VM + m_i_i)),(n_o*PN + (n_i_o*VN + n_i_i))] = g[(m_o*PM + (m_i_o*VM + m_i_i)),(n_o*PN + (n_i_o*VN + n_i_i))-1] + g[(m_o*PM + (m_i_o*VM + m_i_i)),N_plus_1]
    # print(f)
    return f

def compute_root(params):
    # import pdb;pdb.set_trace()
    M,N,h,f = params
    PM = 4
    PN = 16
    VM = 1
    VN = 8
    g = np.zeros((M, N))
    for m in range(M):
        for n in range(N):
            g[m, n] = math.pow(h[m,n], 1.8)
    for m_o in range(M//PM):
        for n_o in range(N//PN):
            for m_i_o in range(PM//VM):
                for n_i_o in range(PN//VN):
                    params = 0,n_o,PN,n_i_o,N,m_o,PM,m_i_o
                    SIMD(f, g, params)
    # print(f)
    return f

def compute_at(params):
    # import pdb;pdb.set_trace()
    M,N,h,f = params
    PM = 4
    PN = 16
    VM = 1
    VN = 8
    for m_o in range(M//PM):
        for n_o in range(N//PN):
            g = np.zeros((PM, PN))
            for mg in range(PM):
                for ng in range(PN):
                    g[mg, ng] = math.pow(h[m_o*PM + mg,n_o*PN + ng], 1.8)
            for m_i_o in range(PM//VM):
                for n_i_o in range(PN//VN):
                    params = 1,n_o,PN,n_i_o,N,m_o,PM,m_i_o
                    SIMD(f, g, params)
    # print(f)
    return f

def compute_tiled(params):
    # import pdb;pdb.set_trace()
    M,N,h,f = params
    PM = 4
    PN = 16
    VM = 1
    VN = 8
    for m_o in range(M//PM):
        for n_o in range(N//PN):
            g = np.zeros((PM, PN))
            # for mg_o in range(PM//VM):
            #     for ng_o in range(PN//VN):
            #         for mg_i in range(VM):
            #             for ng_i in range(VN):
            #                 mg = mg_o*VM + mg_i
            #                 ng = ng_o*VN + ng_i
            #                 g[mg, ng] = math.pow(h[m_o*PM + mg,n_o*PN + ng], 1.8)
            # for m_i_o in range(PM//VM):
            #     for n_i_o in range(PN//VN):
            #         params = 1,n_o,PN,n_i_o,N,m_o,PM,m_i_o
            #         SIMD(f, g, params)

            # for mg_o_o in range(PM//VM//1):
            #     for ng_o_o in range(1):
            #         for mg_o_i in range(1):
            #             for ng_o_i in range(PN//VN):
            #                 for mg_i in range(VM):
            #                     for ng_i in range(VN):
            #                         mg_o = mg_o_o*1 + mg_o_i
            #                         ng_o = ng_o_o*(PN//VN) + ng_o_i
            #                         mg = mg_o*VM + mg_i
            #                         ng = ng_o*VN + ng_i
            #                         g[mg, ng] = math.pow(h[m_o*PM + mg,n_o*PN + ng], 1.8)
            # for m_i_o_o in range(PM//VM//1):
            #     for n_i_o_o in range(1):
            #         for m_i_o_i in range(1):
            #             for n_i_o_i in range(PN//VN):
            #                 m_i_o = m_i_o_o*1 + m_i_o_i
            #                 n_i_o = n_i_o_o*(PN//VN) + n_i_o_i
            #                 params = 1,n_o,PN,n_i_o,N,m_o,PM,m_i_o
            #                 SIMD(f, g, params)

            for m_i_o_o in range(PM//VM//1):
                for n_i_o_o in range(1):
                    for m_i_o_i in range(1):
                        for n_i_o_i in range(PN//VN):
                            for mg_i in range(VM):
                                for ng_i in range(VN):
                                    mg_o = m_i_o_o*1 + m_i_o_i
                                    ng_o = n_i_o_o*(PN//VN) + n_i_o_i
                                    mg = mg_o*VM + mg_i
                                    ng = ng_o*VN + ng_i
                                    g[mg, ng] = math.pow(h[m_o*PM + mg,n_o*PN + ng], 1.8)                    
                    for m_i_o_i in range(1):
                        for n_i_o_i in range(PN//VN):
                            m_i_o = m_i_o_o*1 + m_i_o_i
                            n_i_o = n_i_o_o*(PN//VN) + n_i_o_i
                            params = 1,n_o,PN,n_i_o,N,m_o,PM,m_i_o
                            SIMD(f, g, params)
    # print(f)
    return f


def test():
    u = Util()
    M = 16
    N = 16
    h = np.random.rand(M,N)
    # compute f in row-major order
    f = np.zeros((M, N))
    params = M,N,h,f
    raw_major_order(params)

    # compute f in parallel, vectorized, nested tiles
    ft = np.zeros((M, N))
    params = M,N,h,ft
    parallel_vectorize_tile(params)
    u.check_result(f, ft, 'parallel_vectorize_tile')
    
    # compute g at root
    fcr = np.zeros((M, N))
    params = M,N,h,fcr
    compute_root(params)
    u.check_result(f, fcr, 'compute_root')

    # compute g at tiles of f
    # reduce memory allocate & compute per time
    fca = np.zeros((M, N))
    params = M,N,h,fca
    compute_at(params)
    u.check_result(f, fca, 'compute_at')

    # compute g incrementally within tiles of f
    # reduce compute per time granunarity
    fct = np.zeros((M, N))
    params = M,N,h,fct
    compute_tiled(params)
    u.check_result(f, fct, 'compute_tiled')

    return

if __name__ == '__main__':
    test()