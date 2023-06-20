import torchvision
from transformers import DetrForObjectDetection


def load_detr_model(pretrained_weights="facebook/detr-resnet-50", freeze=False):
    model = DetrForObjectDetection.from_pretrained(pretrained_weights)
    model.config.class_cost = 0.0
    model.config.giou_cost = 0.0
    # model.forward = new_forward
    # model.load_state_dict()

    model.class_labels_classifier.out_features = 1
    model.class_labels_classifier.reset_parameters()
    if freeze:
        # Unfreeze weights
        for param in model.class_labels_classifier.parameters():
            param.requires_grad = True

        for param in model.bbox_predictor.parameters():
            param.requires_grad = True

        for param in model.model.backbone.conv_encoder.model.conv1.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True

    return model


def postprocess_detr_ouput(output):
    boxes = output.pred_boxes
    return torchvision.ops.box_convert(boxes, "cxcywh", "xyxy")
