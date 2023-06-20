from transformers import DetrForObjectDetection


def get_pretrained_detr():
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    # Unfreeze weights
    for params in model.bbox_predictor.parameters():
        params.requires_grad = True

    for params in model.model.backbone.conv_encoder.model.conv1.parameters():
        params.requires_grad = True

    return model


def get_detr_from_checkpoint(checkpoint_data):
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    return model
