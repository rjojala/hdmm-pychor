# Copyright (C) 2019-2021, Tumult Labs Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pychor
import numpy as np
from hdmm import workload, templates
from hdmm.matrix import EkteloMatrix

def get_domain(W):
    if isinstance(W, workload.Kronecker):
        return tuple(Q.shape[1] for Q in W.matrices)
    elif isinstance(W, workload.Weighted):
        return get_domain(W.base)
    elif isinstance(W, workload.VStack):
        return get_domain(W.matrices[0])
    else:
        return W.shape[1]

class HDMM:

    def __init__(self, W, x, eps, seed=0):
        self.domain = get_domain(W)
        self.W = W
        self.x = x
        self.eps = eps
        self.prng = np.random.RandomState(seed)

    def optimize(self, restarts = 25):
        W = self.W
        if type(self.domain) is tuple: # kron or union kron workload
            ns = self.domain

            ps = [max(1, n//16) for n in ns]
            best_strat, best_loss = None, np.inf
            for _ in range(restarts):
                kron = templates.KronPIdentity(ps, ns)
                lossk = kron.optimize(W)
                if lossk < best_loss:
                    best_loss = lossk
                    best_strat = kron

                marg = templates.Marginals(ns)
                lossm = marg.optimize(W)
                if lossm < best_loss:
                    best_loss = lossm
                    best_strat = marg


            # multiplicative factor puts losses on same scale
            self.strategy = best_strat.strategy()
        else:
            n = self.domain
            best_strat, best_loss = None, np.inf
            for _ in range(restarts):
                pid = templates.PIdentity(max(1, n//16), n)
                loss = pid.optimize(W)
                if loss < best_loss:
                    best_loss = loss
                    best_strat = pid
            self.strategy = best_strat.strategy()
           
    def run(self, pychor_parties):
        A = self.strategy
        A1 = A.pinv()
        # Make k copies of the data (self.x)
        # We can do this because the data is 0, so the split will be a vectorized version of the dataset where the length of the splitted vector will be the same as x and it will all be 0 because the data is 0.
        x_parts = [np.copy(self.x) for parties in pychor_parties]
        x_final = []
        delta = self.strategy.sensitivity()
        server = pychor.Party("server")
        # Move into examples 
        for index, x_part in enumerate(x_parts):
            noise = self.prng.laplace(loc=0.0, scale=delta/self.eps, size=A.shape[0])
            # Don't use A_part, use the copies of x
            noisy_dot = pychor_parties[index].constant(A.dot(x_part) + noise)
            noisy_dot.send(src=pychor_parties[index], dest=server)
            pychor.locally(lambda a: x_final.append(np.asarray(a)), noisy_dot.only(server))
        # Do the sum instead of ravel
        self.ans = np.asarray(x_final).sum()
        self.xest = A1.dot(self.ans)
        return self.xest 
 
