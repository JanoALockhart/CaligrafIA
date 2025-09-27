import editdistance

def edit_distance_rate(pred, true):
    return editdistance.eval(pred, true)/len(true)

