# Pytorch Version

def metric(output, label):
    pred = (output >= 0.5)
    truth = (label >= 0.5)
    right = pred.eq(truth)
    acc = right.sum().double().item() / right.numel()
    TP = ( truth* right).sum().double().item()
    TN = (~truth* right).sum().double().item()
    FP = ( truth*~right).sum().double().item()
    FN = (~truth*~right).sum().double().item()
    Pnum = TP+FP
    Nnum = TN+FN
    PR = Pnum / truth.numel()
    TPR = TP/(TP+FN) if TP+FN > 0 else 0
    FPR = FP/(TN+FP) if TN+FP > 0 else 0
    PPrecision = TP/Pnum if Pnum > 0 else 0
    NPrecision = TN/Nnum if Nnum > 0 else 0
    F1 = 2*TP/(2*TP+FP+FN)
    return acc, TPR, FPR, F1, PPrecision, NPrecision, PR
