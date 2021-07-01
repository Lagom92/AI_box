from scipy.spatial import distance

def euclidean_distance(org_landmarks, pred_landmarks):
    total_ed = 0
    for k in range(0, len(org_landmarks), 2):
        ed = distance.euclidean([org_landmarks[k], org_landmarks[k+1]], [pred_landmarks[k], pred_landmarks[k+1]])
        total_ed += ed

    return round(total_ed / 19, 4)