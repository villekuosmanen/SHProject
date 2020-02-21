cimport numpy as np

import numpy as np
from surprise import SVD
from surprise.utils import get_rng
from surprise import PredictionImpossible

class EditableSVD(SVD):
    """An SVD algorithm that allows adding new users without fully retraining the model"""
    def __init__(self, n_factors=100, n_epochs=20, biased=True, init_mean=0,
                 init_std_dev=.1, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 random_state=None, verbose=False):
        SVD.__init__(self, n_factors, n_epochs, biased, init_mean, init_std_dev, 
                    lr_all, reg_all, lr_bu, lr_bi, lr_pu, lr_qi, reg_bu, reg_bi, 
                    reg_pu, reg_qi, random_state, verbose)
            

    def fit_new_user(self, raw_uid, rated_items):
        """
            Fits a new user. Only edits this user's vectors and biases, leaves everything else intact
            rated_items: A dict of (raw) item id mapped to rating given by the new user
        """
        rng = get_rng(self.random_state)
        
        # Increase number of users
        user_inner_id = self.trainset.n_users
        self.trainset.n_users += 1
        self.bu = np.append(self.bu, 0)
        self.pu = np.append(self.pu, 
                rng.normal(self.init_mean, self.init_std_dev, (1, self.n_factors)), 
                axis=0)

        # TODO Add new user to raw and inner ID dicts of trainset
        self.trainset._raw2inner_id_users[raw_uid] = user_inner_id
        if self.trainset._inner2raw_id_users is not None:
            self.trainset._inner2raw_id_users[user_inner_id] = raw_uid
        
        # TODO Add user ratings to trainset (maybe don't need...)

        # user biases
        cdef np.ndarray[np.double_t] bu
        # item biases
        cdef np.ndarray[np.double_t] bi
        # user factors
        cdef np.ndarray[np.double_t, ndim=2] pu
        # item factors
        cdef np.ndarray[np.double_t, ndim=2] qi

        cdef int u, i, f
        cdef double r, err, dot, puf, qif
        cdef double global_mean = self.trainset.global_mean

        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi

        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi

        rng = get_rng(self.random_state)

        bu = self.bu
        bi = self.bi
        pu = self.pu
        qi = self.qi
        u = user_inner_id

        if not self.biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))
            for raw_iid, r in rated_items.items():
                i
                try:
                    i = self.trainset.to_inner_iid(raw_iid)
                except ValueError:
                    continue

                # compute current error
                dot = 0  # <q_i, p_u>
                for f in range(self.n_factors):
                    dot += qi[i, f] * pu[u, f]
                err = r - (global_mean + bu[u] + bi[i] + dot)

                # update biases
                if self.biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    #bi[i] += lr_bi * (err - reg_bi * bi[i])

                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    #qi[i, f] += lr_qi * (err * puf - reg_qi * qif)

        self.bu = bu
        #self.bi = bi
        self.pu = pu
        #self.qi = qi

    def delete_user(self, raw_uid):
        """
            Removes a user. Doesn't remove their effect from the biases of items or other users
            rated_items: A dict of (raw) item id mapped to rating given by the new user
        """
        user_inner_id = self.trainset._raw2inner_id_users[raw_uid]
        self.trainset._raw2inner_id_users.pop(raw_uid)
        if self.trainset._inner2raw_id_users is not None:
            self.trainset._inner2raw_id_users.pop(user_inner_id)
        self.bu = np.delete(self.bu, user_inner_id)
        self.pu = np.delete(self.pu, user_inner_id, axis=0)
        self.trainset.n_users -= 1

    # Override method
    def estimate(self, u, i):
        known_user = u in self.trainset._raw2inner_id_users
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean

            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])

        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unknown.')

        return est

