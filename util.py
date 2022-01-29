

import numpy as np
import pdb
MAX_DIFF_GAP = 1e-4

class Util():

    def check_result(self, res_std, res, string):
        # pdb.set_trace()
        # err = np.sum(np.abs(res - res_std))
        sub = np.abs(res - res_std)
        check = sub.flatten()
        if check.max() > MAX_DIFF_GAP:
            # np.set_printoptions(threshold=np.inf)
            print('>>>>>>>>>>>>fail', string, '<<<<<<<<<<<<<')
            print('res_std:')
            print(res_std)
            print('res:')
            print(res)
            res_diff = res - res_std
            print('diff:')
            print(res_diff)
        else:
            print('>>>>>>>>>>>> success', string, '<<<<<<<<<<<<<')
        return

    def __call__(self):
        return
