def focus(vid):
    r = 0
    for i in range(len(vid)-1):
        r = r + abs(vid[i] - vid[i+1])
    
    b = r > 1
    c = b.astype(int)

    vid = vid * c
    return vid