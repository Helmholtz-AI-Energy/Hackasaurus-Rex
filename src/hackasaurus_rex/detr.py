import torchvision
from transformers import DetrConfig, DetrForObjectDetection, DetrModel


def load_detr_model(pretrained_weights="facebook/detr-resnet-50", freeze=False):
    model = DetrForObjectDetection(base_config)
    # model = DetrForObjectDetection.from_pretrained(pretrained_weights)
    # model.config.class_cost = 0.0
    # model.config.giou_cost = 0.0
    # model.forward = new_forward
    # model.load_state_dict()

    # # model.class_labels_classifier.out_features = 1
    # model.class_labels_classifier.reset_parameters()
    # if freeze:
    #     # Unfreeze weights
    #     for param in model.class_labels_classifier.parameters():
    #         param.requires_grad = True

    #     for param in model.bbox_predictor.parameters():
    #         param.requires_grad = True

    #     for param in model.model.backbone.conv_encoder.model.conv1.parameters():
    #         param.requires_grad = True
    # else:
    #     for param in model.parameters():
    #         param.requires_grad = True

    return model


base_config = DetrConfig(
    use_timm_backbone=False,
    backbone_config=None,
    num_channels=3,
    num_queries=100,
    encoder_layers=6,
    encoder_ffn_dim=2048,
    encoder_attention_heads=4,
    decoder_layers=6,
    decoder_ffn_dim=2048,
    decoder_attention_heads=4,
    encoder_layerdrop=0.0,
    decoder_layerdrop=0.0,
    is_encoder_decoder=True,
    activation_function="gelu",
    d_model=256,
    dropout=0.0,
    attention_dropout=0.0,
    activation_dropout=0.0,
    init_std=0.02,
    init_xavier_std=1.0,
    auxiliary_loss=False,
    position_embedding_type="sine",
    backbone="resnet50",
    use_pretrained_backbone=True,
    dilation=False,
    class_cost=1,
    bbox_cost=2,
    giou_cost=5,
    mask_loss_coefficient=1,
    dice_loss_coefficient=1,
    bbox_loss_coefficient=2,
    giou_loss_coefficient=5,
    eos_coefficient=0.1,
)


def postprocess_detr_ouput(output):
    boxes = output.pred_boxes
    return torchvision.ops.box_convert(boxes, "cxcywh", "xyxy")
