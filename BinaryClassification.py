# Binary Classification Evaluaton Metric, Pytorch Version

def metric(output, label):
    pred = (output >= 0.5)
    truth = (label >= 0.5)
    right = pred.eq(truth)
    acc = right.sum().double().item() / right.numel()   # Accuracy
    TP = ( truth* right).sum().double().item()
    TN = (~truth* right).sum().double().item()
    FP = ( truth*~right).sum().double().item()
    FN = (~truth*~right).sum().double().item()
    Pnum = TP+FP
    Nnum = TN+FN
    PR = Pnum / truth.numel()                   # Positive Rate
    TPR = TP/(TP+FN) if TP+FN > 0 else 0        # True Positive Rate
    FPR = FP/(TN+FP) if TN+FP > 0 else 0        # False Positive Rate
    Recall = TP/(TP+FN)                         # Recall
    PPrecision = TP/Pnum if Pnum > 0 else 0     # Positive Precision
    NPrecision = TN/Nnum if Nnum > 0 else 0     # Negative Precision
    F1 = 2*TP/(2*TP+FP+FN)                      # F1 score
    return acc, TPR, FPR, F1, Recall, PPrecision, NPrecision, PR



# Training Part
acc, TPR, FPR, F1, Recall, PPrecision, NPrecision, PR = accuracy(pred, b_label.float())
print("Epoch %2d Iteration: %d BCELoss: %f"%(epoch, it, loss.item()))
print(" - Acc: %.2f PR: %.2f TPR: %.2f FPR: %.2f F1: %.2f Recall: %.2f PPrecision: %.2f NPrecision: %.2f\n"%(acc, PR, TPR, FPR, F1, Recall, PPrecision, NPrecision))

# Testing Part
acc, TPR, FPR, F1, Recall, PPrecision, NPrecision, PR = accuracy(torch.tensor(test_pred), torch.tensor(test_label))
print("Epoch %2d Test Acc: %.2f PR: %.2f"%(epoch, acc, PR))
print(" - TPR: %.2f FPR: %.2f F1: %.2f Recall: %.2f PPrecision: %.2f NPrecision: %.2f\n"%(TPR, FPR, F1, Recall, PPrecision, NPrecision))













