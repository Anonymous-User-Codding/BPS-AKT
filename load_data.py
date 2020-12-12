import numpy as np
import math


class DATA(object):
    def __init__(self, n_skill, seqlen, separate_char, name="data"):
        self.separate_char = separate_char
        self.n_skill = n_skill
        self.seqlen = seqlen


    def load_data(self, path):
        file_data = open(path, 'r')
        s_data = []
        sa_data = []
        t_data = []
        f_data = []
        x_data = []
        y_data = []
        idx_data = []
        for lineID, line in enumerate(file_data):
            line = line.strip()
            if lineID % 7 == 0:
                learner_id = lineID//3
            if lineID % 7 == 1:
                S = line.split(self.separate_char)
                if len(S[len(S)-1]) == 0:
                    S = S[:-1]
            if lineID % 7 == 2:
                X = line.split(self.separate_char)
                if len(X[len(X)-1]) == 0:
                    X = X[:-1]
            if lineID % 7 == 3:
                Y = line.split(self.separate_char)
                if len(Y[len(Y)-1]) == 0:
                    Y = Y[:-1]
            if lineID % 7 == 4:
                T = line.split(self.separate_char)
                if len(T[len(T)-1]) == 0:
                    T = T[:-1]
            if lineID % 7 == 5:
                F = line.split(self.separate_char)
                if len(F[len(F)-1]) == 0:
                    F = F[:-1]
            elif lineID % 7 == 6:
                A = line.split(self.separate_char)
                if len(A[len(A)-1]) == 0:
                    A = A[:-1]

                # start split the data
                n_split = 1
                if len(S) > self.seqlen:
                    n_split = math.floor(len(S) / self.seqlen)
                    if len(S) % self.seqlen:
                        n_split = n_split + 1
                for k in range(n_split):
                    s_seq = []
                    sa_seq = []
                    t_seq = []
                    f_seq = []
                    x_seq = []
                    y_seq = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(S[i]) > 0:
                            sa = int(S[i]) + int(A[i]) * self.n_skill
                            s_seq.append(int(S[i]))
                            sa_seq.append(sa)
                            t_seq.append(int(T[i]))
                            f_seq.append(int(F[i]))
                            x_seq.append(int(X[i]))
                            y_seq.append(int(Y[i]))
                        else:
                            print(S[i])
                    s_data.append(s_seq)
                    sa_data.append(sa_seq)
                    t_data.append(t_seq)
                    f_data.append(f_seq)
                    x_data.append(x_seq)
                    y_data.append(y_seq)

                    idx_data.append(learner_id)
        file_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        s_dataArray = np.zeros((len(s_data), self.seqlen))
        for j in range(len(s_data)):
            dat = s_data[j]
            s_dataArray[j, :len(dat)] = dat

        sa_dataArray = np.zeros((len(sa_data), self.seqlen))
        for j in range(len(sa_data)):
            dat = sa_data[j]
            sa_dataArray[j, :len(dat)] = dat

        t_dataArray = np.zeros((len(t_data), self.seqlen))
        for j in range(len(t_data)):
            dat = t_data[j]
            t_dataArray[j, :len(dat)] = dat
        f_dataArray = np.zeros((len(f_data), self.seqlen))
        for j in range(len(f_data)):
            dat = f_data[j]
            f_dataArray[j, :len(dat)] = dat
        x_dataArray = np.zeros((len(x_data), self.seqlen))
        for j in range(len(x_data)):
            dat = x_data[j]
            x_dataArray[j, :len(dat)] = dat
        y_dataArray = np.zeros((len(y_data), self.seqlen))
        for j in range(len(y_data)):
            dat = y_data[j]
            y_dataArray[j, :len(dat)] = dat
        return s_dataArray, sa_dataArray, np.asarray(idx_data), t_dataArray, f_dataArray, x_dataArray, y_dataArray

    def load_test_data(self, path):
        file_data = open(path, 'r')
        s_data = []
        sa_data = []
        t_data = []
        f_data = []
        x_data = []
        y_data = []
        idx_data = []
        test_s_num = 0
        for lineID, line in enumerate(file_data):
            line = line.strip()
            if lineID % 7 == 0:
                learner_id = lineID//3
            if lineID % 7 == 1:
                S = line.split(self.separate_char)
                if len(S[len(S)-1]) == 0:
                    S = S[:-1]
                test_s_num += len(S)
            if lineID % 7 == 2:
                X = line.split(self.separate_char)
                if len(X[len(X)-1]) == 0:
                    X = X[:-1]
            if lineID % 7 == 3:
                Y = line.split(self.separate_char)
                if len(Y[len(Y)-1]) == 0:
                    Y = Y[:-1]
            if lineID % 7 == 4:
                T = line.split(self.separate_char)
                if len(T[len(T)-1]) == 0:
                    T = T[:-1]
            if lineID % 7 == 5:
                F = line.split(self.separate_char)
                if len(F[len(F)-1]) == 0:
                    F = F[:-1]
            elif lineID % 7 == 6:
                A = line.split(self.separate_char)
                if len(A[len(A)-1]) == 0:
                    A = A[:-1]

                # start split the data
                n_split = 1
                if len(S) > self.seqlen:
                    n_split = math.floor(len(S) / self.seqlen)
                    if len(S) % self.seqlen:
                        n_split = n_split + 1
                for k in range(n_split):
                    s_seq = []
                    sa_seq = []
                    t_seq = []
                    f_seq = []
                    x_seq = []
                    y_seq = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(S[i]) > 0:
                            sa = int(S[i]) + int(A[i]) * self.n_skill
                            s_seq.append(int(S[i]))
                            sa_seq.append(sa)
                            t_seq.append(int(T[i]))
                            f_seq.append(int(F[i]))
                            x_seq.append(int(X[i]))
                            y_seq.append(int(Y[i]))
                        else:
                            print(S[i])
                    s_data.append(s_seq)
                    sa_data.append(sa_seq)
                    t_data.append(t_seq)
                    f_data.append(f_seq)
                    x_data.append(x_seq)
                    y_data.append(y_seq)

                    idx_data.append(learner_id)
        file_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        s_dataArray = np.zeros((len(s_data), self.seqlen))
        for j in range(len(s_data)):
            dat = s_data[j]
            s_dataArray[j, :len(dat)] = dat

        sa_dataArray = np.zeros((len(sa_data), self.seqlen))
        for j in range(len(sa_data)):
            dat = sa_data[j]
            sa_dataArray[j, :len(dat)] = dat

        t_dataArray = np.zeros((len(t_data), self.seqlen))
        for j in range(len(t_data)):
            dat = t_data[j]
            t_dataArray[j, :len(dat)] = dat
        f_dataArray = np.zeros((len(f_data), self.seqlen))
        for j in range(len(f_data)):
            dat = f_data[j]
            f_dataArray[j, :len(dat)] = dat
        x_dataArray = np.zeros((len(x_data), self.seqlen))
        for j in range(len(x_data)):
            dat = x_data[j]
            x_dataArray[j, :len(dat)] = dat
        y_dataArray = np.zeros((len(y_data), self.seqlen))
        for j in range(len(y_data)):
            dat = y_data[j]
            y_dataArray[j, :len(dat)] = datq
        return s_dataArray, sa_dataArray, np.asarray(idx_data),\
               t_dataArray, f_dataArray, x_dataArray, y_dataArray, test_s_num

class EID_DATA(object):
    def __init__(self, n_skill,  seqlen, separate_char, name="data"):
        self.separate_char = separate_char
        self.seqlen = seqlen
        self.n_skill = n_skill

    def load_data(self, path):
        file_data = open(path, 'r')
        s_data = []
        sa_data = []
        e_data = []
        t_data = []
        f_data = []
        x_data = []
        y_data = []
        for lineID, line in enumerate(file_data):
            line = line.strip()
            if lineID % 8 == 0:
                learner_id = lineID//3
            if lineID % 8 == 1:
                S = line.split(self.separate_char)
                if len(S[len(S)-1]) == 0:
                    S = S[:-1]
            if lineID % 8 == 2:
                E = line.split(self.separate_char)
                if len(E[len(E)-1]) == 0:
                    E = E[:-1]
            if lineID % 8 == 3:
                X = line.split(self.separate_char)
                if len(X[len(X)-1]) == 0:
                    X = X[:-1]
            if lineID % 8 == 4:
                Y = line.split(self.separate_char)
                if len(Y[len(Y)-1]) == 0:
                    Y = Y[:-1]
            if lineID % 8 == 5:
                T = line.split(self.separate_char)
                if len(T[len(T)-1]) == 0:
                    T = T[:-1]
            if lineID % 8 == 6:
                F = line.split(self.separate_char)
                if len(F[len(F)-1]) == 0:
                    F = F[:-1]
            elif lineID % 8 == 7:
                A = line.split(self.separate_char)
                if len(A[len(A)-1]) == 0:
                    A = A[:-1]

                # start split the data
                n_split = 1
                if len(S) > self.seqlen:
                    n_split = math.floor(len(S) / self.seqlen)
                    if len(S) % self.seqlen:
                        n_split = n_split + 1
                for k in range(n_split):
                    s_seq = []
                    sa_seq = []
                    e_seq = []
                    t_seq = []
                    f_seq = []
                    x_seq = []
                    y_seq = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(S[i]) > 0:
                            sa = int(S[i]) + int(A[i]) * self.n_skill
                            s_seq.append(int(S[i]))
                            e_seq.append(int(E[i]))
                            t_seq.append(int(T[i]))
                            f_seq.append(int(F[i]))
                            x_seq.append(int(X[i]))
                            y_seq.append(int(Y[i]))
                            sa_seq.append(sa)
                        else:
                            print(S[i])
                    s_data.append(s_seq)
                    sa_data.append(sa_seq)
                    e_data.append(e_seq)
                    t_data.append(t_seq)
                    f_data.append(f_seq)
                    x_data.append(x_seq)
                    y_data.append(y_seq)

        file_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        s_dataArray = np.zeros((len(s_data), self.seqlen))
        for j in range(len(s_data)):
            dat = s_data[j]
            s_dataArray[j, :len(dat)] = dat

        sa_dataArray = np.zeros((len(sa_data), self.seqlen))
        for j in range(len(sa_data)):
            dat = sa_data[j]
            sa_dataArray[j, :len(dat)] = dat

        e_dataArray = np.zeros((len(e_data), self.seqlen))
        for j in range(len(e_data)):
            dat = e_data[j]
            e_dataArray[j, :len(dat)] = dat

        t_dataArray = np.zeros((len(t_data), self.seqlen))
        for j in range(len(t_data)):
            dat = t_data[j]
            t_dataArray[j, :len(dat)] = dat

        f_dataArray = np.zeros((len(f_data), self.seqlen))
        for j in range(len(f_data)):
            dat = f_data[j]
            f_dataArray[j, :len(dat)] = dat

        x_dataArray = np.zeros((len(x_data), self.seqlen))
        for j in range(len(x_data)):
            dat = x_data[j]
            x_dataArray[j, :len(dat)] = dat

        y_dataArray = np.zeros((len(y_data), self.seqlen))
        for j in range(len(y_data)):
            dat = y_data[j]
            y_dataArray[j, :len(dat)] = dat

        return s_dataArray, sa_dataArray, e_dataArray,\
               t_dataArray, f_dataArray, x_dataArray, y_dataArray

    def load_test_data(self, path):
        file_data = open(path, 'r')
        s_data = []
        sa_data = []
        e_data = []
        t_data = []
        f_data = []
        x_data = []
        y_data = []
        test_s_num = 0
        for lineID, line in enumerate(file_data):
            line = line.strip()
            if lineID % 8 == 0:
                learner_id = lineID//3
            if lineID % 8 == 1:
                S = line.split(self.separate_char)
                if len(S[len(S)-1]) == 0:
                    S = S[:-1]
                test_s_num += len(S)
            if lineID % 8 == 2:
                E = line.split(self.separate_char)
                if len(E[len(E)-1]) == 0:
                    E = E[:-1]
            if lineID % 8 == 3:
                X = line.split(self.separate_char)
                if len(X[len(X)-1]) == 0:
                    X = X[:-1]
            if lineID % 8 == 4:
                Y = line.split(self.separate_char)
                if len(Y[len(Y)-1]) == 0:
                    Y = Y[:-1]
            if lineID % 8 == 5:
                T = line.split(self.separate_char)
                if len(T[len(T)-1]) == 0:
                    T = T[:-1]
            if lineID % 8 == 6:
                F = line.split(self.separate_char)
                if len(F[len(F)-1]) == 0:
                    F = F[:-1]
            elif lineID % 8 == 7:
                A = line.split(self.separate_char)
                if len(A[len(A)-1]) == 0:
                    A = A[:-1]

                # start split the data
                n_split = 1
                if len(S) > self.seqlen:
                    n_split = math.floor(len(S) / self.seqlen)
                    if len(S) % self.seqlen:
                        n_split = n_split + 1
                for k in range(n_split):
                    s_seq = []
                    e_seq = []
                    t_seq = []
                    f_seq = []
                    x_seq = []
                    y_seq = []
                    sa_seq = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(S[i]) > 0:
                            sa = int(S[i]) + int(A[i]) * self.n_skill
                            s_seq.append(int(S[i]))
                            e_seq.append(int(E[i]))
                            t_seq.append(int(T[i]))
                            f_seq.append(int(F[i]))
                            x_seq.append(int(X[i]))
                            y_seq.append(int(Y[i]))
                            sa_seq.append(sa)
                        else:
                            print(S[i])
                    s_data.append(s_seq)
                    sa_data.append(sa_seq)
                    e_data.append(e_seq)
                    t_data.append(t_seq)
                    f_data.append(f_seq)
                    x_data.append(x_seq)
                    y_data.append(y_seq)

        file_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        s_dataArray = np.zeros((len(s_data), self.seqlen))
        for j in range(len(s_data)):
            dat = s_data[j]
            s_dataArray[j, :len(dat)] = dat

        sa_dataArray = np.zeros((len(sa_data), self.seqlen))
        for j in range(len(sa_data)):
            dat = sa_data[j]
            sa_dataArray[j, :len(dat)] = dat

        e_dataArray = np.zeros((len(e_data), self.seqlen))
        for j in range(len(e_data)):
            dat = e_data[j]
            e_dataArray[j, :len(dat)] = dat

        t_dataArray = np.zeros((len(t_data), self.seqlen))
        for j in range(len(t_data)):
            dat = t_data[j]
            t_dataArray[j, :len(dat)] = dat

        f_dataArray = np.zeros((len(f_data), self.seqlen))
        for j in range(len(f_data)):
            dat = f_data[j]
            f_dataArray[j, :len(dat)] = dat

        x_dataArray = np.zeros((len(x_data), self.seqlen))
        for j in range(len(x_data)):
            dat = x_data[j]
            x_dataArray[j, :len(dat)] = dat

        y_dataArray = np.zeros((len(y_data), self.seqlen))
        for j in range(len(y_data)):
            dat = y_data[j]
            y_dataArray[j, :len(dat)] = dat

        return s_dataArray, sa_dataArray, e_dataArray,\
               t_dataArray, f_dataArray, x_dataArray, y_dataArray, test_s_num
