import numpy as np
import scipy as sp

def delta_hedge(Swapts,Q,T_s,t):
    T_sl = T_s.copy()
    T_sl = [T_sl[i] for i in range(len(T_sl)-1) if T_sl[i+1] > t] + [T_sl[-1]]

    swapts_hedge_amount = np.zeros(len(Swapts))
    Q_Hedge_Amount = np.zeros(len(Q))

    if len(T_sl) <= 2:
        return (swapts_hedge_amount,Q_Hedge_Amount)

    else: 
        Q_vec = [Q[T] for T in T_sl] # Q from T_a to T_b
        croppedSwapts = [Swapts[T] for T in T_sl[:-1]] # Swaptions from T_a to T_b-1

        SwapHedgeAmount = [(Q_vec[k-1] - Q_vec[k])  for k in range(1,len(T_sl))] 

        Swapts0 = croppedSwapts + [0.0]                         # Zero padded swaptions so that we can loop over a sum of swaptions
        QHedgeAmount = [0] + [Swapts0[k] - Swapts0[k-1] for k in range(1,len(T_sl))]   # How much to hedge in Q starting at indx a and ending at Q(T_b-1)

        # Adjustment term
        if False: #t > T_s[0]: 
            SwapHedgeAmount[0] = 0
            SwapHedgeAmount[1] = (1 - Q_vec[2])
            
            QHedgeAmount[1] = 0
            QHedgeAmount[2] = -Swapts0[1] + Swapts0[2]
            Q_Hedge_Amount[0] = Swapts0[1]# Just for value balance, since Q0 never moves this has no effect on the efficacy of the hedge, but balances the books s.t the delta hedge costs exactly twice the value of the CVA
        
        # The Qvector and Swapts vector are at the latest possible position, so
        offsetS = len(swapts_hedge_amount) - len(SwapHedgeAmount)
        offsetQ = len(Q_Hedge_Amount) - len(QHedgeAmount)
        swapts_hedge_amount[offsetS:] = SwapHedgeAmount
        Q_Hedge_Amount[offsetQ:] = QHedgeAmount

    return (swapts_hedge_amount, Q_Hedge_Amount)