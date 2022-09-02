import numpy as np

def build_tps_fps_weight_min_sparse(target, weight):
    from scipy.sparse import bsr_array
    from sparse import COO

    expand_target = np.expand_dims(target, axis=1)
    expand_target = COO(expand_target)
    
    expand_weight = np.expand_dims(weight, axis=0)
    
    ret = expand_target * expand_weight
    ret = np.min(ret, axis=1)
    return ret.todense()


def build_tps_fps_weight_min(target, weight):
    from scipy.sparse import bsr_array
    from sparse import COO

    expand_target = np.expand_dims(target, axis=1)
    
    expand_weight = np.expand_dims(weight, axis=0)

    ret = expand_target * expand_weight
    ret = np.min(ret, axis=1)
    return ret

if __name__ == "__main__":
    target = np.random.randn(128, 32)
    weight = np.random.randn(32, 32)
    
    sparse = build_tps_fps_weight_min_sparse(target, weight)
    nonsparse = build_tps_fps_weight_min(target, weight)
    
    import ipdb; ipdb.set_trace()