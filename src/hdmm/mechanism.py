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
           
    def run(self, pychor_parties, x_parts):
        A = self.strategy
        A1 = A.pinv()
        x_final = []
        GAMMA = 1000
        THETA = 0.01
        delta = 10e-5
        DELTA = self.strategy.sensitivity()
        print(delta)
        print(self.eps)
        print(np.log(1/delta))
        rho = (np.sqrt(np.log(1 / delta) + self.eps) - np.sqrt(np.log(1 / delta)))**2
        # p from line 62 from base_config.py in distributed hdmm
        P = 2**31 - 1
        server = pychor.Party("server")
        num_clients = len(pychor_parties) + 1
        # Possible problems:
        # Not using discrete gaussian
        # The noise scale from the paper != the noise in the repo (agent bellnew)
        # Can't tell if p or rho
        # Does N_z() represent variance or SD?
        # Isn't noise now dependent on data?
        std_dev = (GAMMA * delta) / np.sqrt(2*(1-THETA)*num_clients*rho)
        # std_dev = GAMMA*delta**2/(num_clients/2)
        # std_dev = (GAMMA**2*delta**2)/(2*(1-THETA)*num_clients*rho)
        # std_dev = delta/self.eps
        for index, x_part in enumerate(x_parts):
            x_local = pychor_parties[index].constant(x_part)
            # noise = self.prng.normal(loc=0.0, scale=delta/(self.eps*std_dev), size=A.shape[0])
            # encoded = pychor_parties[index].constant(A.dot(x_part) + noise)
            meas_vec = pychor.locally(lambda data: A.dot(data), x_local)
            meas_vec_scaled = pychor.locally(lambda meas: (meas * GAMMA).astype(int), meas_vec)
            noisy_val = pychor.locally(lambda meas, scaled: scaled + (self.prng.normal(loc=0.0, scale=std_dev, size=len(meas)).astype(int)), meas_vec, meas_vec_scaled)
            # print(noisy_val)
            encoded = pychor.locally(lambda val: val % P, noisy_val)

            encoded.send(src=pychor_parties[index], dest=server)
            pychor.locally(lambda a: x_final.append(np.asarray(a)), encoded.only(server))
        self.ans = np.asarray(x_final).sum(axis=0)
        self.xest = A1.dot(self.ans)
        return self.xest 
 
