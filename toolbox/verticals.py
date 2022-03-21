def thickness(midpoints):
    n = len(midpoints)
    thick = [2*midpoints[0]]
    lower = thick[0]
    for point in midpoints[1:]:
        thick.append(2*(point-lower))
        lower = lower + thick[-1]
    return thick

def midpoint(thickness):
    midpoints = [thickness[0]/2.0]
    lower = thickness[0]
    for thick in thickness[1:]:
        midpoints.append(lower + thick/2.0)
        lower += thick
    return midpoints

def boundaries(thickness):
    bound = [0.0]
    for thick in thickness:
        bound.append(bound[-1] + thick)
    return bound
    

#mid = [25,100,250,525,1000,2000,3000,4500,6000,7500,9000,10500,12000]
#print(thickness(mid))
#print(midpoint(thickness(mid)))
#print(boundaries(thickness(mid)))
