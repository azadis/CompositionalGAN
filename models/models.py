# ========================================================
# Compositional GAN
# Models to be loaded
# By Samaneh Azadi
# ========================================================


def create_model(opt):
    model = None
    if opt.model == 'objCompose':
        if opt.dataset_mode == 'comp_decomp_aligned':
            from .objCompose_supervised_model import objComposeSuperviseModel
            model = objComposeSuperviseModel()
        elif opt.dataset_mode == 'comp_decomp_unaligned':
            from .objCompose_unsupervised_model import objComposeUnsuperviseModel
            model = objComposeUnsuperviseModel()
        elif opt.dataset_mode == 'baseline':
            from .objCompose_baseline import objComposeBaselineModel
            model = objComposeBaselineModel()
        elif opt.dataset_mode == 'baseline_unaligned':
            from .objCompose_cycleGAN_baseline import objComposeBaselineCycleModel
            model = objComposeBaselineCycleModel()
        else:
            raise ValueError("Model [%s] not recognized." % opt.model)

    elif opt.model == 'AFN':
        assert(opt.dataset_mode == 'AFN')
        from .AFN_model import AFNModel
        model = AFNModel()

    elif opt.model == 'AFNCompose':
        assert(opt.dataset_mode == 'AFNCompose')
        from .AFN_compose_model import AFNComposeModel
        model = AFNComposeModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
