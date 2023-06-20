from transformers import DetrForObjectDetection


def load_detr_model(pretrained_weights="facebook/detr-resnet-50", freeze=False):
    model = DetrForObjectDetection.from_pretrained(pretrained_weights)
    model.load_state_dict()

    if freeze:
        # Unfreeze weights
        for param in model.bbox_predictor.parameters():
            param.requires_grad = True

        for param in model.model.backbone.conv_encoder.model.conv1.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True

    return model
