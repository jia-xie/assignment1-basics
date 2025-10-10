import math

def get_lr_cosine_schedule(t: float, lr_max:float, lr_min:float, Tw:float, Tc:float):
    assert Tw < Tc, "The number of warmup iterations must be smaller than cosine annealing iterations"
    assert lr_min < lr_max, "Minimum learning rate must be smaller than maximum learning rate"
    assert Tw > 0 and lr_min > 0, "Learning rate and number of iterations must be positive"
    
    if t < Tw:
        return t/Tw*lr_max
    elif t<=Tc:
        return lr_min + 0.5*(1+math.cos(math.pi*(t-Tw)/(Tc-Tw)))*(lr_max-lr_min)
    else:
        return lr_min