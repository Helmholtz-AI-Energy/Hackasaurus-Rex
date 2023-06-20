from ultralytics.yolo.utils.loss import BboxLoss


def get_loss():
    return BboxLoss(reg_max=1, use_dfl=False)
