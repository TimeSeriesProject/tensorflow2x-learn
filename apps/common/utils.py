
def calcu_acc(predict, t_label):
    assert(len(predict) == len(t_label))
    if len(predict) == 0:
        return 0
    tp = 0
    for idx, value in enumerate(predict):
        if value == t_label[idx]:
            tp += 1
    return tp/len(predict)
