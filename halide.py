# https://zhuanlan.zhihu.com/p/462306283
import numpy as np
from sklearn import preprocessing
from sympy import im
import time
import pdb
from util import *

class halide:
    debug = 0
    def __init__(self, debug = 0) -> None:
        self.debug = debug
    
    def blur(self, image, bv, param):
        '''
        bh = blur horizontal
        bv = blur vertical
        '''
        H,W = param
        bh = np.zeros((H,W))
        # pdb.set_trace()
        for y in range(H):
            for x in range(W):
            # for x in range(1,W-1,1):
                bh[x,y] = (image[x-1,y] + image[x,y] + image[x+1 if x+1 < W else 0,y])//3

        for y in range(H):
        # for y in range(1,H-1,1):
            for x in range(W):
                bv[x,y] = (bh[x,y-1] + bh[x,y] + bh[x,y+1 if y+1 < H else 0])//3

        if self.debug:
            print('image:')
            print(image)
            print('bh:')
            print(bh)
            print('bv:')
            print(bv)
    
    def fast_blur(self, image, bv, param):
        '''
        bh = blur horizontal
        bv = blur vertical
        yt = y tile
        xt = x tile
        '''
        H,W = param
        one_third = 0.333
        
        i_f = image.flatten()
        for yt in range(0,H,32):
            # bh = np.zeros(((256//8)*(32+2))) # (32*34)
            bh = np.zeros((H,W)) # (32,32)
            for xt in range(0,W,256):
                # bhptr = 0
                # for y in range(-1, 32+1, 1):
                for y in range(0, H, 1):
                    # inptr = xt * yt + y
                    for x in range(0, W, 8):
                        x_start = xt + x
                        x_end   = x_start + 8
                        y_start = yt + y        
                        a = image[y_start-1, x_start:x_end]
                        b = image[y_start,   x_start:x_end]
                        if y_start+1 >= H:
                            # pdb.set_trace()
                            y_start = y_start-H
                        c = image[y_start+1 if y_start+1 < H else 0, x_start:x_end]
                        sum = a + b + c
                        # avg = sum * one_third
                        avg = sum // 3
                        bh[y_start, x_start:x_end] = avg
                        # inptr += 8
                # pdb.set_trace()
                # for y in range(0, H, 1):
                #     for x in range(0, W, 8):
                for y in range(0, H, 8):
                    for x in range(0, W, 1):
                        x_start = xt + x
                        y_start = yt + y
                        y_end   = y_start + 8
                        a = bh[y_start:y_end, x_start-1]
                        b = bh[y_start:y_end, x_start]
                        c = bh[y_start:y_end, x_start+1 if x_start+1 < W else 0]
                        sum = a + b + c
                        # avg = sum * one_third
                        avg = sum // 3
                        bv[y_start:y_end, x_start] = avg

        # pdb.set_trace()
        if self.debug:
            print('image:')
            print(image)
            print('bh:')
            print(bh)
            print('bv:')
            print(bv)

def test():
    '''
    preprocessing
    TODO: open image.yuv
    ''' 
    np.set_printoptions(threshold=np.inf)

    # H,W = 256,256
    # H,W = 32,32
    H,W = 16,16
    param = H,W
    np.random.seed(0)
    image = np.random.randint(10, size=(H,W))
    '''
    bh = a horizontal blur
    bv = a vertical blur
    '''     
    
    # h = halide(debug=1)
    h = halide()
    u = Util()


    bv_1 = np.zeros((H,W))
    time_start =time.time()
    h.blur(image, bv_1, param)
    time_end =time.time()
    print('1.blur duration:', (time_end - time_start))

    time_start =time.time()
    bv_2 = np.zeros((H,W))
    h.fast_blur(image, bv_2, param)
    time_end =time.time()
    print('2.fast_blur duration:', (time_end - time_start))

    u.check_result(bv_1, bv_2, 'blur')

    
if __name__ == '__main__':
    test()
        