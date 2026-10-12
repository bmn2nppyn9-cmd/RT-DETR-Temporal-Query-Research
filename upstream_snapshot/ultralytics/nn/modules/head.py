# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model head modules."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils import NOT_MACOS14
from ultralytics.utils.tal import dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import TORCH_1_11, fuse_conv_and_bn, smart_inference_mode

from .block import DFL, SAVPE, BNContrastiveHead, ContrastiveHead, Proto, Proto26, RealNVP, Residual, SwiGLUFFN
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = (
    "OBB",
    "Classify",
    "Detect",
    "Pose",
    "RTDETRDecoder",
    "Segment",
    "SemanticSegment",
    "YOLOEDetect",
    "YOLOESegment",
    "v10Detect",
)


class Detect(nn.Module):
    """YOLO Detect head for object detection models.

    This class implements the detection head used in YOLO models for predicting bounding boxes and class probabilities.
    It supports both training and inference modes, with optional end-to-end detection capabilities.

    Attributes:
        dynamic (bool): Force grid reconstruction.
        export (bool): Export mode flag.
        format (str): Export format.
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum detections per image.
        shape (tuple): Input shape.
        anchors (torch.Tensor): Anchor points.
        strides (torch.Tensor): Feature map strides.
        legacy (bool): Backward compatibility for v3/v5/v8/v9/v11 models.
        xyxy (bool): Output format, xyxy or xywh.
        nc (int): Number of classes.
        nl (int): Number of detection layers.
        reg_max (int): DFL channels.
        no (int): Number of outputs per anchor.
        stride (torch.Tensor): Strides computed during build.
        cv2 (nn.ModuleList): Convolution layers for box regression.
        cv3 (nn.ModuleList): Convolution layers for classification.
        dfl (nn.Module): Distribution Focal Loss layer.
        one2one_cv2 (nn.ModuleList): One-to-one convolution layers for box regression.
        one2one_cv3 (nn.ModuleList): One-to-one convolution layers for classification.

    Methods:
        forward: Perform forward pass and return predictions.
        bias_init: Initialize detection head biases.
        decode_bboxes: Decode bounding boxes from predictions.
        postprocess: Post-process model predictions.

    Examples:
        Create a detection head for 80 classes
        >>> detect = Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = detect(x)
    """

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    max_det = 300  # max_det
    agnostic_nms = False
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    xyxy = False  # xyxy or xywh output

    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO detection layer with specified number of classes and channels.

        Args:
            nc (int): Number of classes.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)

    @property
    def end2end(self):
        """Checks if the model has one2one for v3/v5/v8/v9/v11 backward compatibility."""
        return getattr(self, "_end2end", True) and hasattr(self, "one2one")

    @end2end.setter
    def end2end(self, value):
        """Override the end-to-end detection mode."""
        self._end2end = value

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]  # batch size
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

        Args:
            x (dict[str, torch.Tensor]): Dictionary of predictions from detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
        """
        # Inference path
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get decoded boxes based on anchors and strides."""
        shape = x["feats"][0].shape  # BCHW
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape

        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return dbox

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):  # from
            a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / self.stride[i]) ** 2
            )  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):  # from
                a[-1].bias.data[:] = 2.0  # box
                b[-1].bias.data[: self.nc] = math.log(
                    5 / self.nc / (640 / self.stride[i]) ** 2
                )  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """Decode bounding boxes from predictions."""
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not self.end2end and not self.xyxy,
            dim=1,
        )

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x1, y1, x2, y2, class_probs].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x1, y1, x2, y2, max_class_prob, class_index].
        """
        boxes, scores = preds.split([4, self.nc], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores, conf], dim=-1)

    def get_topk_index(self, scores: torch.Tensor, max_det: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get top-k indices from scores.

        Args:
            scores (torch.Tensor): Scores tensor with shape (batch_size, num_anchors, num_classes).
            max_det (int): Maximum detections per image.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): Top scores, class indices, and filtered indices.
        """
        batch_size, anchors, nc = scores.shape  # i.e. shape(16,8400,80)
        # Use max_det directly during export for TensorRT compatibility (requires k to be constant),
        # otherwise use min(max_det, anchors) for safety with small inputs during Python inference
        k = max_det if self.export else min(max_det, anchors)
        if self.agnostic_nms:
            scores, labels = scores.max(dim=-1, keepdim=True)
            scores, indices = scores.topk(k, dim=1)
            labels = labels.gather(1, indices)
            return scores, labels, indices
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(k)
        idx = ori_index[torch.arange(batch_size)[..., None], index // nc]  # original index
        return scores[..., None], (index % nc)[..., None].float(), idx

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = None


class Segment(Detect):
    """YOLO Segment head for segmentation models.

    This class extends the Detect head to include mask prediction capabilities for instance segmentation tasks.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv4 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a segmentation head
        >>> segment = Segment(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, mask_head=self.one2one_cv4)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with mask coefficients."""
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, mask_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and mask coefficients."""
        preds = super().forward_head(x, box_head, cls_head)
        if mask_head is not None:
            bs = x[0].shape[0]  # batch size
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nm) with last dimension
                format [x1, y1, x2, y2, class_probs, mask_coefficient].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + nm) and last
                dimension format [x1, y1, x2, y2, max_class_prob, class_index, mask_coefficient].
        """
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = None


class Segment26(Segment):
    """YOLO26 Segment head for segmentation models.

    This class extends the Segment head with Proto26 for mask prediction in instance segmentation tasks.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto26): Prototype generation module.
        cv4 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a segmentation head
        >>> segment = Segment26(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, nm, npr, reg_max, end2end, ch)
        self.proto = Proto26(ch, self.npr, self.nm, nc)  # protos

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = Detect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x)  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = (
                    tuple(p.detach() for p in proto) if isinstance(proto, tuple) else proto.detach()
                )
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def fuse(self) -> None:
        """Remove the one2many head and extra part of proto module for inference optimization."""
        super().fuse()
        if hasattr(self.proto, "fuse"):
            self.proto.fuse()


class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models.

    This class extends the Detect head to include oriented bounding box prediction with rotation angles.

    Attributes:
        ne (int): Number of extra parameters.
        cv4 (nn.ModuleList): Convolution layers for angle prediction.
        angle (torch.Tensor): Predicted rotation angles.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        decode_bboxes: Decode rotated bounding boxes.

    Examples:
        Create an OBB detection head
        >>> obb = OBB(nc=80, ne=1, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb(x)
    """

    def __init__(self, nc: int = 80, ne: int = 1, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`.

        Args:
            nc (int): Number of classes.
            ne (int): Number of extra parameters.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, angle_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, angle_head=self.one2one_cv4)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with rotation angles."""
        # For decode_bboxes convenience
        self.angle = x["angle"]
        preds = super()._inference(x)
        return torch.cat([preds, x["angle"]], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, angle_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and angles."""
        preds = super().forward_head(x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]  # batch size
            angle = torch.cat(
                [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )  # OBB theta logits
            angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
            preds["angle"] = angle
        return preds

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + ne) with last dimension
                format [x, y, w, h, class_probs, angle].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 7) and last
                dimension format [x, y, w, h, max_class_prob, class_index, angle].
        """
        boxes, scores, angle = preds.split([4, self.nc, self.ne], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        angle = angle.gather(dim=1, index=idx.repeat(1, 1, self.ne))
        return torch.cat([boxes, scores, conf, angle], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = None


class OBB26(OBB):
    """YOLO26 OBB detection head for detection with rotation models. This class extends the OBB head with modified angle
    processing that outputs raw angle predictions without sigmoid transformation, compared to the original
    OBB class.

    Attributes:
        ne (int): Number of extra parameters.
        cv4 (nn.ModuleList): Convolution layers for angle prediction.
        angle (torch.Tensor): Predicted rotation angles.

    Methods:
        forward_head: Concatenate and return predicted bounding boxes, class probabilities, and raw angles.

    Examples:
        Create an OBB26 detection head
        >>> obb26 = OBB26(nc=80, ne=1, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb26(x)
    """

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, angle_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and raw angles."""
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]  # batch size
            angle = torch.cat(
                [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )  # OBB theta logits (raw output without sigmoid transformation)
            preds["angle"] = angle
        return preds


class Pose(Detect):
    """YOLO Pose head for keypoints models.

    This class extends the Detect head to include keypoint prediction capabilities for pose estimation tasks.

    Attributes:
        kpt_shape (tuple): Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible).
        nk (int): Total number of keypoint values.
        cv4 (nn.ModuleList): Convolution layers for keypoint prediction.

    Methods:
        forward: Perform forward pass through YOLO model and return predictions.
        kpts_decode: Decode keypoints from predictions.

    Examples:
        Create a pose detection head
        >>> pose = Pose(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize YOLO network with default parameters and Convolutional Layers.

        Args:
            nc (int): Number of classes.
            kpt_shape (tuple): Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible).
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, pose_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, pose_head=self.one2one_cv4)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with keypoints."""
        preds = super()._inference(x)
        return torch.cat([preds, self.kpts_decode(x["kpts"])], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, pose_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and keypoints."""
        preds = super().forward_head(x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]  # batch size
            preds["kpts"] = torch.cat([pose_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nk) with last dimension
                format [x1, y1, x2, y2, class_probs, keypoints].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + self.nk) and
                last dimension format [x1, y1, x2, y2, max_class_prob, class_index, keypoints].
        """
        boxes, scores, kpts = preds.split([4, self.nc, self.nk], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        kpts = kpts.gather(dim=1, index=idx.repeat(1, 1, self.nk))
        return torch.cat([boxes, scores, conf, kpts], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = None

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions."""
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Pose26(Pose):
    """YOLO26 Pose head for keypoints models.

    This class extends the Pose head with normalizing flow for keypoint prediction in pose estimation tasks.

    Attributes:
        kpt_shape (tuple): Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible).
        nk (int): Total number of keypoint values.
        cv4 (nn.ModuleList): Convolution layers for keypoint prediction.

    Methods:
        forward: Perform forward pass through YOLO model and return predictions.
        kpts_decode: Decode keypoints from predictions.

    Examples:
        Create a pose detection head
        >>> pose = Pose26(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize YOLO network with default parameters and Convolutional Layers.

        Args:
            nc (int): Number of classes.
            kpt_shape (tuple): Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible).
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)
        self.flow_model = RealNVP()

        c4 = max(ch[0] // 4, kpt_shape[0] * (kpt_shape[1] + 2))
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3)) for x in ch)

        self.cv4_kpts = nn.ModuleList(nn.Conv2d(c4, self.nk, 1) for _ in ch)
        self.nk_sigma = kpt_shape[0] * 2  # sigma_x, sigma_y for each keypoint
        self.cv4_sigma = nn.ModuleList(nn.Conv2d(c4, self.nk_sigma, 1) for _ in ch)

        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)
            self.one2one_cv4_kpts = copy.deepcopy(self.cv4_kpts)
            self.one2one_cv4_sigma = copy.deepcopy(self.cv4_sigma)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(
            box_head=self.cv2,
            cls_head=self.cv3,
            pose_head=self.cv4,
            kpts_head=self.cv4_kpts,
            kpts_sigma_head=self.cv4_sigma,
        )

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            pose_head=self.one2one_cv4,
            kpts_head=self.one2one_cv4_kpts,
            kpts_sigma_head=self.one2one_cv4_sigma,
        )

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        pose_head: torch.nn.Module,
        kpts_head: torch.nn.Module,
        kpts_sigma_head: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and keypoints."""
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]  # batch size
            features = [pose_head[i](x[i]) for i in range(self.nl)]
            preds["kpts"] = torch.cat([kpts_head[i](features[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
            if self.training:
                preds["kpts_sigma"] = torch.cat(
                    [kpts_sigma_head[i](features[i]).view(bs, self.nk_sigma, -1) for i in range(self.nl)], 2
                )
        return preds

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        super().fuse()
        self.cv4_kpts = self.cv4_sigma = self.flow_model = self.one2one_cv4_sigma = None

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions."""
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            # NCNN fix
            a = (y[:, :, :2] + self.anchors) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] + self.anchors[0]) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] + self.anchors[1]) * self.strides
            return y


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2).

    This class implements a classification head that transforms feature maps into class predictions.

    Attributes:
        export (bool): Export mode flag.
        conv (Conv): Convolutional layer for feature transformation.
        pool (nn.AdaptiveAvgPool2d): Global average pooling layer.
        drop (nn.Dropout): Dropout layer for regularization.
        linear (nn.Linear): Linear layer for final classification.

    Methods:
        forward: Perform forward pass on input feature maps.

    Examples:
        Create a classification head
        >>> classify = Classify(c1=1024, c2=1000)
        >>> x = torch.randn(1, 1024, 20, 20)
        >>> output = classify(x)
    """

    export = False  # export mode

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output classes.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor | tuple:
        """Perform forward pass on input feature maps."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to incorporate text embeddings for enhanced semantic understanding in
    object detection tasks.

    Attributes:
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        bias_init: Initialize detection head biases.

    Examples:
        Create a WorldDetect head
        >>> world_detect = WorldDetect(nc=80, embed=512, with_bn=False, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = world_detect(x, text)
    """

    def __init__(
        self,
        nc: int = 80,
        embed: int = 512,
        with_bn: bool = False,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        """Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max=reg_max, end2end=end2end, ch=ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> dict[str, torch.Tensor] | tuple:
        """Concatenate and return predicted bounding boxes and class probabilities."""
        feats = [xi.clone() for xi in x]  # save original features for anchor generation
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        bs = x[0].shape[0]
        x_cat = torch.cat([xi.view(bs, self.no, -1) for xi in x], 2)
        boxes, scores = x_cat.split((self.reg_max * 4, self.nc), 1)
        preds = dict(boxes=boxes, scores=scores, feats=feats)
        if self.training:
            return preds
        y = self._inference(preds)
        return y if self.export else (y, preds)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class LRPCHead(nn.Module):
    """Lightweight Region Proposal and Classification Head for efficient object detection.

    This head combines region proposal filtering with classification to enable efficient detection with dynamic
    vocabulary support.

    Attributes:
        vocab (nn.Module): Vocabulary/classification layer.
        pf (nn.Module): Proposal filter module.
        loc (nn.Module): Localization module.
        enabled (bool): Whether the head is enabled.

    Methods:
        conv2linear: Convert a 1x1 convolutional layer to a linear layer.
        forward: Process classification and localization features to generate detection proposals.

    Examples:
        Create an LRPC head
        >>> vocab = nn.Conv2d(256, 80, 1)
        >>> pf = nn.Conv2d(256, 1, 1)
        >>> loc = nn.Conv2d(256, 4, 1)
        >>> head = LRPCHead(vocab, pf, loc, enabled=True)
    """

    def __init__(self, vocab: nn.Module, pf: nn.Module, loc: nn.Module, enabled: bool = True):
        """Initialize LRPCHead with vocabulary, proposal filter, and localization components.

        Args:
            vocab (nn.Module): Vocabulary/classification module.
            pf (nn.Module): Proposal filter module.
            loc (nn.Module): Localization module.
            enabled (bool): Whether to enable the head functionality.
        """
        super().__init__()
        self.vocab = self.conv2linear(vocab) if enabled else vocab
        self.pf = pf
        self.loc = loc
        self.enabled = enabled

    @staticmethod
    def conv2linear(conv: nn.Conv2d) -> nn.Linear:
        """Convert a 1x1 convolutional layer to a linear layer."""
        assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
        linear = nn.Linear(conv.in_channels, conv.out_channels)
        linear.weight.data = conv.weight.view(conv.out_channels, -1).data
        linear.bias.data = conv.bias.data
        return linear

    def forward(self, cls_feat: torch.Tensor, loc_feat: torch.Tensor, conf: float) -> tuple[tuple, torch.Tensor]:
        """Process classification and localization features to generate detection proposals."""
        if self.enabled:
            pf_score = self.pf(cls_feat)[0, 0].flatten(0)
            mask = pf_score.sigmoid() > conf
            cls_feat = cls_feat.flatten(2).transpose(-1, -2)
            cls_feat = self.vocab(cls_feat[:, mask] if conf else cls_feat * mask.unsqueeze(-1).int())
            return self.loc(loc_feat), cls_feat.transpose(-1, -2), mask
        else:
            cls_feat = self.vocab(cls_feat)
            loc_feat = self.loc(loc_feat)
            return (
                loc_feat,
                cls_feat.flatten(2),
                torch.ones(cls_feat.shape[2] * cls_feat.shape[3], device=cls_feat.device, dtype=torch.bool),
            )


class YOLOEDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to support text-guided detection with enhanced semantic understanding
    through text embeddings and visual prompt embeddings.

    Attributes:
        is_fused (bool): Whether the model is fused for inference.
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.
        reprta (Residual): Residual block for text prompt embeddings.
        savpe (SAVPE): Spatial-aware visual prompt embeddings module.
        embed (int): Embedding dimension.

    Methods:
        fuse: Fuse text features with model weights for efficient inference.
        get_tpe: Get text prompt embeddings with normalization.
        get_vpe: Get visual prompt embeddings with spatial awareness.
        forward_lrpc: Process features with fused text embeddings for prompt-free model.
        forward: Process features with class prompt embeddings to generate detections.
        bias_init: Initialize biases for detection heads.

    Examples:
        Create a YOLOEDetect head
        >>> yoloe_detect = YOLOEDetect(nc=80, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> cls_pe = torch.randn(1, 80, 512)
        >>> outputs = yoloe_detect([*x, cls_pe])
    """

    is_fused = False

    def __init__(
        self, nc: int = 80, embed: int = 512, with_bn: bool = False, reg_max=16, end2end=False, ch: tuple = ()
    ):
        """Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        c3 = max(ch[0], min(self.nc, 100))
        assert c3 <= embed
        assert with_bn
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, embed, 1),
                )
                for x in ch
            )
        )
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)
        if end2end:
            self.one2one_cv3 = copy.deepcopy(self.cv3)  # overwrite with new cv3
            self.one2one_cv4 = copy.deepcopy(self.cv4)

        self.reprta = Residual(SwiGLUFFN(embed, embed))
        self.savpe = SAVPE(ch, c3, embed)
        self.embed = embed

    @smart_inference_mode()
    def fuse(self, txt_feats: torch.Tensor = None):
        """Fuse text features with model weights for efficient inference."""
        if txt_feats is None:  # means eliminate one2many branch
            self.cv2 = self.cv3 = self.cv4 = None
            return
        if self.is_fused:
            return

        assert not self.training
        txt_feats = txt_feats.to(torch.float32).squeeze(0)
        if self.cv3 and self.cv4:
            self._fuse_tp(txt_feats, self.cv3, self.cv4)
        if self.end2end:
            self._fuse_tp(txt_feats, self.one2one_cv3, self.one2one_cv4)
        del self.reprta
        self.reprta = nn.Identity()
        self.is_fused = True

    def _fuse_tp(self, txt_feats: torch.Tensor, cls_head: torch.nn.Module, bn_head: torch.nn.Module) -> None:
        """Fuse text prompt embeddings with model weights for efficient inference."""
        for cls_h, bn_h in zip(cls_head, bn_head):
            assert isinstance(cls_h, nn.Sequential)
            assert isinstance(bn_h, BNContrastiveHead)
            conv = cls_h[-1]
            assert isinstance(conv, nn.Conv2d)
            logit_scale = bn_h.logit_scale
            bias = bn_h.bias
            norm = bn_h.norm

            t = txt_feats * logit_scale.exp()
            conv: nn.Conv2d = fuse_conv_and_bn(conv, norm)

            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data

            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bias

            conv = (
                nn.Conv2d(
                    conv.in_channels,
                    w.shape[0],
                    kernel_size=1,
                )
                .requires_grad_(False)
                .to(conv.weight.device)
            )

            conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            conv.bias.data.copy_(b1 + b2)
            cls_h[-1] = conv

            bn_h.fuse()

    def get_tpe(self, tpe: torch.Tensor | None) -> torch.Tensor | None:
        """Get text prompt embeddings with normalization."""
        return None if tpe is None else F.normalize(self.reprta(tpe), dim=-1, p=2)

    def get_vpe(self, x: list[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
        """Get visual prompt embeddings with spatial awareness."""
        if vpe.shape[1] == 0:  # no visual prompt embeddings
            return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device)
        if vpe.ndim == 4:  # (B, N, H, W)
            vpe = self.savpe(x, vpe)
        assert vpe.ndim == 3  # (B, N, D)
        return vpe

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with class prompt embeddings to generate detections."""
        if hasattr(self, "lrpc"):  # for prompt-free inference
            return self.forward_lrpc(x[:3])
        return super().forward(x)

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with fused text embeddings to generate detections for prompt-free model."""
        boxes, scores, index = [], [], []
        bs = x[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv3
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                0 if self.export and not self.dynamic else getattr(self, "conf", 0.001),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        preds = dict(boxes=torch.cat(boxes, 2), scores=torch.cat(scores, 2), feats=x, index=torch.cat(index))
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _get_decode_boxes(self, x):
        """Decode predicted bounding boxes for inference."""
        dbox = super()._get_decode_boxes(x)
        if hasattr(self, "lrpc"):
            dbox = dbox if self.export and not self.dynamic else dbox[..., x["index"]]
        return dbox

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, contrastive_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, contrastive_head=self.one2one_cv4)

    def forward_head(self, x, box_head, cls_head, contrastive_head):
        """Concatenates and returns predicted bounding boxes, class probabilities, and contrastive scores."""
        assert len(x) == 4, f"Expected 4 features including 3 feature maps and 1 text embeddings, but got {len(x)}."
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]  # batch size
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        self.nc = x[-1].shape[1]
        scores = torch.cat(
            [contrastive_head[i](cls_head[i](x[i]), x[-1]).reshape(bs, self.nc, -1) for i in range(self.nl)], dim=-1
        )
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        return dict(boxes=boxes, scores=scores, feats=x[:3])

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for i, (a, b, c) in enumerate(
            zip(self.one2many["box_head"], self.one2many["cls_head"], self.one2many["contrastive_head"])
        ):
            a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[:] = 0.0
            c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b, c) in enumerate(
                zip(self.one2one["box_head"], self.one2one["cls_head"], self.one2one["contrastive_head"])
            ):
                a[-1].bias.data[:] = 2.0  # box
                b[-1].bias.data[:] = 0.0
                c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)


class YOLOESegment(YOLOEDetect):
    """YOLO segmentation head with text embedding capabilities.

    This class extends YOLOEDetect to include mask prediction capabilities for instance segmentation tasks with
    text-guided semantic understanding.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv5 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a YOLOESegment head
        >>> yoloe_segment = YOLOESegment(nc=80, nm=32, npr=256, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = yoloe_segment([*x, text])
    """

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        """Initialize YOLOESegment with class count, mask parameters, and embedding dimensions.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv5, contrastive_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            mask_head=self.one2one_cv5,
            contrastive_head=self.one2one_cv4,
        )

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with fused text embeddings to generate detections for prompt-free model."""
        boxes, scores, index = [], [], []
        bs = x[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv3
        cv5 = self.cv5 if not self.end2end else self.one2one_cv5
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                0 if self.export and not self.dynamic else getattr(self, "conf", 0.001),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        mc = torch.cat([cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        index = torch.cat(index)
        preds = dict(
            boxes=torch.cat(boxes, 2),
            scores=torch.cat(scores, 2),
            feats=x,
            index=index,
            mask_coefficient=mc * index.int() if self.export and not self.dynamic else mc[..., index],
        )
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with mask coefficients."""
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        mask_head: torch.nn.Module,
        contrastive_head: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and mask coefficients."""
        preds = super().forward_head(x, box_head, cls_head, contrastive_head)
        if mask_head is not None:
            bs = x[0].shape[0]  # batch size
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nm) with last dimension
                format [x1, y1, x2, y2, class_probs, mask_coefficient].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + nm) and last
                dimension format [x1, y1, x2, y2, max_class_prob, class_index, mask_coefficient].
        """
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self, txt_feats: torch.Tensor = None):
        """Fuse text features with model weights for efficient inference."""
        super().fuse(txt_feats)
        if txt_feats is None:  # means eliminate one2many branch
            self.cv5 = None
            if hasattr(self.proto, "fuse"):
                self.proto.fuse()
            return


class YOLOESegment26(YOLOESegment):
    """YOLOE-style segmentation head module using Proto26 for mask generation.

    This class extends the YOLOESegment functionality to include segmentation capabilities by integrating a Proto26
    generation module and convolutional layers to predict mask coefficients.

    Args:
        nc (int): Number of classes. Defaults to 80.
        nm (int): Number of masks. Defaults to 32.
        npr (int): Number of prototype channels. Defaults to 256.
        embed (int): Embedding dimensionality. Defaults to 512.
        with_bn (bool): Whether to use Batch Normalization. Defaults to False.
        reg_max (int): Maximum number of DFL channels. Defaults to 16.
        end2end (bool): Whether to use end-to-end detection mode. Defaults to False.
        ch (tuple[int, ...]): Input channels for each scale.

    Attributes:
        nm (int): Number of segmentation masks.
        npr (int): Number of prototype channels.
        proto (Proto26): Prototype generation module for segmentation.
        cv5 (nn.ModuleList): Convolutional layers for generating mask coefficients from features.
        one2one_cv5 (nn.ModuleList, optional): Deep copy of cv5 for end-to-end detection branches.
    """

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        """Initialize YOLOESegment26 with class count, mask parameters, and embedding dimensions."""
        YOLOEDetect.__init__(self, nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto26(ch, self.npr, self.nm, nc)  # protos

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = YOLOEDetect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto([xi.detach() for xi in x], return_semantic=False)  # mask protos

        if isinstance(preds, dict):  # training and validating during training
            if self.end2end and not hasattr(self, "lrpc"):  # not prompt-free
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)


class RTDETRDecoder(nn.Module):
    """Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.

    Attributes:
        export (bool): Export mode flag.
        hidden_dim (int): Dimension of hidden layers.
        nhead (int): Number of heads in multi-head attention.
        nl (int): Number of feature levels.
        nc (int): Number of classes.
        num_queries (int): Number of query points.
        num_decoder_layers (int): Number of decoder layers.
        input_proj (nn.ModuleList): Input projection layers for backbone features.
        decoder (DeformableTransformerDecoder): Transformer decoder module.
        denoising_class_embed (nn.Embedding): Class embeddings for denoising.
        num_denoising (int): Number of denoising queries.
        label_noise_ratio (float): Label noise ratio for training.
        box_noise_scale (float): Box noise scale for training.
        learnt_init_query (bool): Whether to learn initial query embeddings.
        tgt_embed (nn.Embedding): Target embeddings for queries.
        query_pos_head (MLP): Query position head.
        enc_output (nn.Sequential): Encoder output layers.
        enc_score_head (nn.Linear): Encoder score prediction head.
        enc_bbox_head (MLP): Encoder bbox prediction head.
        dec_score_head (nn.ModuleList): Decoder score prediction heads.
        dec_bbox_head (nn.ModuleList): Decoder bbox prediction heads.

    Methods:
        forward: Run forward pass and return bounding box and classification scores.

    Examples:
        Create an RTDETRDecoder
        >>> decoder = RTDETRDecoder(nc=80, ch=(512, 1024, 2048), hd=256, nq=300)
        >>> x = [torch.randn(1, 512, 64, 64), torch.randn(1, 1024, 32, 32), torch.randn(1, 2048, 16, 16)]
        >>> outputs = decoder(x)
    """

    export = False  # export mode
    shapes = []
    anchors = torch.empty(0)
    valid_mask = torch.empty(0)
    dynamic = False

    # Temporal query propagation probe (eval-only, disabled by default). Declared at class level so that
    # checkpoints pickled before this feature existed still resolve the attributes on unpickle (instance
    # __dict__ restored from an old pickle bypasses __init__). When disabled every code path is identical
    # to the original implementation; nothing below touches training.
    temporal_enabled = False  # master switch for previous-frame query propagation
    num_temporal_queries = 75  # propagated queries appended after the current encoder queries
    temporal_motion_alpha = 0.0  # 0 -> box_t = box_(t-1); 0.5 -> + 0.5 * velocity
    temporal_motion_match_iou = 0.3  # class-agnostic IoU for velocity association
    _temporal_state = None  # multi-depth previous-frame state, see _update_temporal_state()
    _encoder_query_override = None  # control group: more current-frame encoder queries (A1)
    _output_topk_override = None  # final detections cap (defaults to num_queries)
    _last_topk_meta = None  # per-output query_idx / scores / source / source_depth / injection_layer
    _last_decoder_query_count = 0  # actual decoder input width of the last forward (diagnostics)
    _last_n_current = 0  # current-frame encoder query count of the last forward (diagnostics)

    # Query-depth probe (eval-only, disabled by default). Layered on top of the temporal probe:
    #   return_all_layers: export per-layer decoder states after each forward (QD-O oracle export)
    #   temporal_source_depth / temporal_injection_layer / temporal_fuse_after: staged temporal injection
    #     (QD-T scan). temporal_fuse_after bypasses the decoder entirely (T3O lower-bound reference)
    #   time_decoder_cuda: CUDA-Event timing around the decoder call (latency reporting)
    return_all_layers = False
    _last_layer_states = None  # {layer: (hidden, box, logits, refer_in)} detached, from the last forward
    _last_layer_widths = None  # {layer: query width} of the last decoder pass when layer states were collected
    temporal_source_depth = 3  # 0..3: propagate S0 (encoder) .. S3 (L3) query states of the previous frame
    temporal_injection_layer = 0  # 0/1/2: decoder layer before which temporal queries are appended
    temporal_fuse_after = False  # append previous-frame L3 results after the decoder (no decoder pass)
    time_decoder_cuda = False
    _last_decoder_ms = 0.0

    # Query-depth schedule probe (QD-P, eval-only, disabled by default): staged query exit by per-layer width
    # schedule. Exited queries freeze their layer output; later layers really run on the surviving subset.
    exit_schedule = None  # e.g. [300, 150, 75]: width entering each layer; first entry must equal query count
    exit_policy = 'confidence'  # confidence | stability | confidence+stability | random | gt_oracle
    _exit_masks = None  # per-frame precomputed {layer: (B, Q_i) bool} masks for random / gt_oracle (caller-set)
    _last_exit_ann = None  # (B, 300): -1 = ran to eval layer, i = exited after layer i
    _last_exit_widths = None  # [q0, width_l2, width_l3, ...] of the last forward when an exit plan was active
    _last_dec_boxes = None  # merged per-query boxes of the exit path (diagnostics)
    time_model_cuda = False
    _last_model_ms = 0.0

    # B1a depth-aligned temporal query propagation (training feature, branch depth-aligned-propagation-b1a).
    # b1a_struct_override: class-level dict set BEFORE model construction, keys:
    #   nq / ndl / nd: structural overrides (num queries, decoder layers, denoising) so one yaml serves all
    #       B1a configs; nd=0 cleanly disables denoising (get_cdn_group returns all-None).
    #   eval_topk: postprocess output cap for the trained model (e.g. 325-query control still outputs 300).
    #   temporal: None for controls, else dict(M, source_depth, injection_layer) enabling the pair-training
    #       path (extract_pair_state / forward_train_pair) and creating temporal_adapter / temporal_src_embed.
    # Declared at class level so checkpoints pickled without these attrs still unpickle (instance __dict__
    # bypasses __init__; missing attrs resolve on the class).
    b1a_struct_override = None
    b1a_eval_topk = None  # instance attr set from override; class default for old checkpoints
    b1a_temporal_spec = None

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,  # hidden dim
        nq: int = 300,  # num queries
        ndp: int = 4,  # num decoder points
        nh: int = 8,  # num head
        ndl: int = 6,  # num decoder layers
        d_ffn: int = 1024,  # dim of feedforward
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        # Training args
        nd: int = 100,  # num denoising
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
    ):
        """Initialize the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes.
            ch (tuple): Channels in the backbone feature maps.
            hd (int): Dimension of hidden layers.
            nq (int): Number of query points.
            ndp (int): Number of decoder points.
            nh (int): Number of heads in multi-head attention.
            ndl (int): Number of decoder layers.
            d_ffn (int): Dimension of the feed-forward networks.
            dropout (float): Dropout rate.
            act (nn.Module): Activation function.
            eval_idx (int): Evaluation index.
            nd (int): Number of denoising.
            label_noise_ratio (float): Label noise ratio.
            box_noise_scale (float): Box noise scale.
            learnt_init_query (bool): Whether to learn initial query embeddings.
        """
        super().__init__()
        # B1a structural overrides take precedence over the signature defaults (class-level dict).
        ovr = type(self).b1a_struct_override or {}
        nq, ndl, nd = ovr.get('nq', nq), ovr.get('ndl', ndl), ovr.get('nd', nd)
        self.b1a_eval_topk = ovr.get('eval_topk')
        self.b1a_temporal_spec = ovr.get('temporal')
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

        # B1a temporal source adaptation (only for temporal configs). Both start as exact identities so an
        # untrained model degenerates to the frozen-injection probe behavior (regression anchor R6).
        if self.b1a_temporal_spec is not None:
            self.temporal_adapter = nn.Linear(hd, hd)
            nn.init.eye_(self.temporal_adapter.weight)
            nn.init.zeros_(self.temporal_adapter.bias)
            self.temporal_src_embed = nn.Parameter(torch.zeros(1, 1, hd))

    # ------------------------------------------------------------------ temporal probe API (eval-only)
    def set_temporal_enabled(self, flag: bool):
        """Enable or disable previous-frame query propagation. Disabling also clears the temporal state."""
        self.temporal_enabled = bool(flag)
        if not flag:
            self._temporal_state = None

    def reset_temporal_state(self):
        """Drop any propagated state. Must be called at scene boundaries and after config changes."""
        self._temporal_state = None

    def set_query_overrides(
        self,
        encoder_queries: int | None = None,
        temporal_queries: int | None = None,
        output_topk: int | None = None,
        motion_alpha: float | None = None,
        motion_match_iou: float | None = None,
    ):
        """Configure query-count concepts independently (eval-only probe).

        Args:
            encoder_queries: Current-frame encoder top-k query count (default num_queries=300; 375 for control A1).
            temporal_queries: Number of propagated previous-frame queries appended after the current ones.
            output_topk: Final detection cap in postprocess (default num_queries=300).
            motion_alpha: Position extrapolation weight; 0.0 disables motion (box_t = box_(t-1)).
            motion_match_iou: Class-agnostic IoU threshold used when estimating per-query velocity.
        """
        if encoder_queries is not None:
            self._encoder_query_override = int(encoder_queries)
        if temporal_queries is not None:
            self.num_temporal_queries = int(temporal_queries)
        if output_topk is not None:
            self._output_topk_override = int(output_topk)
        if motion_alpha is not None:
            self.temporal_motion_alpha = float(motion_alpha)
        if motion_match_iou is not None:
            self.temporal_motion_match_iou = float(motion_match_iou)

    def set_temporal_config(
        self,
        source_depth: int | None = None,
        injection_layer: int | None = None,
        fuse_after: bool | None = None,
    ):
        """Configure the staged temporal injection probe (eval-only, QD-T).

        Args:
            source_depth: Which previous-frame state to propagate: 0 = encoder content/proposal (S0),
                1/2/3 = L1/L2/L3 hidden + refined box (S1..S3).
            injection_layer: Decoder layer before which the temporal queries are appended (0/1/2).
            fuse_after: If True, previous-frame L3 results are appended after the decoder instead of being
                injected into it (T3O lower-bound reference).
        """
        if source_depth is not None:
            self.temporal_source_depth = int(source_depth)
        if injection_layer is not None:
            self.temporal_injection_layer = int(injection_layer)
        if fuse_after is not None:
            self.temporal_fuse_after = bool(fuse_after)

    def set_exit_plan(self, schedule: list[int] | None = None, policy: str | None = None):
        """Configure the staged query-exit probe (eval-only, QD-P).

        Args:
            schedule: Per-layer query widths, e.g. [300, 150, 75] for widths entering L1/L2/L3. After each layer
                i, queries exit until the next layer's width is met; the first entry must equal the current
                query count. None disables the probe.
            policy: How exits are chosen — 'confidence', 'stability', 'confidence+stability' (decided online
                inside the decoder), or 'random' / 'gt_oracle' (masks precomputed per frame by the caller in
                self._exit_masks; the model itself never sees GT).
        """
        if schedule is None:
            self.exit_schedule = None  # disable the probe
            self._exit_masks = None
        else:
            self.exit_schedule = [int(x) for x in schedule]
        if policy is not None:
            self.exit_policy = str(policy)

    def _select_exit_mask(
        self,
        i: int,
        output: torch.Tensor,
        refined_bbox: torch.Tensor,
        cls_logits: torch.Tensor,
        refer_in: torch.Tensor,
    ) -> torch.Tensor:
        """Choose which queries exit after decoder layer i (eval-only probe, online policies).

        Returns a bool mask (B, Q) aligned with the current query slots. The number of exits is fixed by
        self.exit_schedule (the width entering layer i+1); the ranking depends on exit_policy:
          confidence: keep the highest max-class scores;
          stability: keep the queries whose boxes moved least since the layer input (small refinement step);
          confidence+stability: keep the highest score * (1 - box_delta).
        """
        n_next = self.exit_schedule[i + 1]
        if n_next >= cls_logits.shape[1]:
            return torch.zeros_like(cls_logits[..., 0], dtype=torch.bool)
        if self.exit_policy == 'stability':
            rank = -(refined_bbox - refer_in).abs().mean(-1)  # (B, Q)
        else:
            score = cls_logits.sigmoid().amax(-1)
            if self.exit_policy == 'confidence':
                rank = score
            else:  # confidence+stability
                rank = score * (1.0 - (refined_bbox - refer_in).abs().mean(-1))
        keep = rank.topk(n_next, dim=1).indices  # (B, n_next)
        mask = torch.ones_like(rank, dtype=torch.bool)
        mask.scatter_(1, keep, False)
        return mask

    def temporal_status(self) -> dict:
        """Report the probe configuration and whether a previous-frame state is currently held."""
        return {
            'temporal_enabled': self.temporal_enabled,
            'has_state': self._temporal_state is not None,
            'num_temporal_queries': self.num_temporal_queries,
            'encoder_queries': self._encoder_query_override or self.num_queries,
            'output_topk': self._output_topk_override or self.num_queries,
            'motion_alpha': self.temporal_motion_alpha,
            'source_depth': self.temporal_source_depth,
            'injection_layer': self.temporal_injection_layer,
            'fuse_after': self.temporal_fuse_after,
            'exit_schedule': self.exit_schedule,
            'exit_policy': self.exit_policy,
            'last_decoder_query_count': self._last_decoder_query_count,
        }

    def _gather_temporal_state(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Gather the temporal (embed_add, refer_add) pair from the configured source depth.

        embed_add carries the stored hidden/content tensor (1, M, 256); refer_add carries the stored box in
        sigmoid space (1, M, 4) because mid-layer injection happens after the decoder has already sigmoid()'ed
        the initial refer_bbox. Returns None when no previous-frame state is held (scene first frame).
        """
        st = self._temporal_state
        if st is None:
            return None
        boxes = st['boxes'][self.temporal_source_depth]
        if self.temporal_motion_alpha:
            boxes = boxes + self.temporal_motion_alpha * st['velocity']
        boxes = boxes.clamp(1e-4, 1.0 - 1e-4)
        embed_out = st['hidden'][self.temporal_source_depth]
        # B1a: a trained temporal model carries a source adapter/embedding. The eval injection path must apply
        # the same transformation as the training path, otherwise record/eval would not match training.
        if getattr(self, 'temporal_adapter', None) is not None:
            embed_out = self.temporal_adapter(embed_out) + self.temporal_src_embed
        return embed_out, boxes

    @staticmethod
    def _cxcywh_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Pairwise IoU between cxcywh boxes, a: (B, N, 4), b: (B, M, 4) -> (B, N, M)."""
        a_xyxy = torch.cat([a[..., :2] - a[..., 2:] / 2, a[..., :2] + a[..., 2:] / 2], -1)
        b_xyxy = torch.cat([b[..., :2] - b[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2], -1)
        lt = torch.maximum(a_xyxy[:, :, None, :2], b_xyxy[:, None, :, :2])
        rb = torch.minimum(a_xyxy[:, :, None, 2:], b_xyxy[:, None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        area_a = a[..., 2] * a[..., 3]
        area_b = b[..., 2] * b[..., 3]
        return inter / (area_a[:, :, None] + area_b[:, None, :] - inter).clamp(min=1e-9)

    def _update_temporal_state(
        self,
        layer_states: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        embed0: torch.Tensor,
        enc_bboxes: torch.Tensor,
    ) -> None:
        """Store the previous frame's query states at every depth (S0..S3) for the same top-M query slots.

        Selection is by the L3 per-query max class score (score3) restricted to the first n_current (encoder)
        queries, so propagated queries never re-enter the state and every source depth carries the exact same
        query slots (source depth is isolated from selection). S0 is the encoder content/proposal before L1;
        S1..S3 are the per-layer hidden/refined-box outputs; logits3 keeps the full L3 class logits (T3O fuse).
        Velocity is a class-agnostic IoU association against the previous state's S3 boxes and is only used
        when temporal_motion_alpha != 0.
        """
        n = self._last_n_current
        h3, b3, l3, _ = layer_states[2]
        h3, b3, l3 = h3[:, :n], b3[:, :n], l3[:, :n]
        scores = l3.sigmoid().amax(-1)  # (B, n)
        m = min(self.num_temporal_queries, n)
        top = scores.topk(m, dim=1).indices  # (B, m)
        bs = embed0.shape[0]
        bi = torch.arange(bs, device=embed0.device).unsqueeze(-1)

        def pick(x: torch.Tensor) -> torch.Tensor:
            return x[bi, top].detach().clone()

        h1, b1, _, _ = layer_states[0]
        h2, b2, _, _ = layer_states[1]
        new_boxes3 = pick(b3)
        velocity = torch.zeros_like(new_boxes3)
        prev = self._temporal_state
        if prev is not None:
            iou = self._cxcywh_iou(new_boxes3, prev['boxes'][3])  # (B, m, M_prev)
            best = iou.argmax(-1)
            ok = iou.gather(-1, best.unsqueeze(-1)).squeeze(-1) >= self.temporal_motion_match_iou
            delta = new_boxes3 - prev['boxes'][3][bi, best]
            velocity = torch.where(ok.unsqueeze(-1), delta, torch.zeros_like(delta)).detach()
        self._temporal_state = {
            'idx': top.detach().clone(),
            'hidden': {
                0: pick(embed0),
                1: pick(h1[:, :n]),
                2: pick(h2[:, :n]),
                3: pick(h3),
            },
            'boxes': {
                0: pick(enc_bboxes),
                1: pick(b1[:, :n]),
                2: pick(b2[:, :n]),
                3: new_boxes3,
            },
            'logits3': pick(l3),
            'scores3': pick(scores),
            'velocity': velocity,
        }

    def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor:
        """Run the forward pass of the module, returning bounding box and classification scores for the input.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.
            batch (dict, optional): Batch information for training.

        Returns:
            outputs (tuple | torch.Tensor): During training, returns a tuple of bounding boxes, scores, and other
                metadata. During inference, returns a tensor of shape (bs, num_queries, 6) containing bounding boxes,
                confidence scores, and class labels.
        """
        from ultralytics.models.utils.ops import get_cdn_group

        t0m = t1m = None
        if self.time_model_cuda and torch.cuda.is_available() and not self.training:
            t0m, t1m = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0m.record()

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        if self.training and (
            self.temporal_enabled
            or self._encoder_query_override
            or self._output_topk_override
            or self.return_all_layers
            or self.time_decoder_cuda
            or self.time_model_cuda
            or self.exit_schedule is not None
        ):
            raise RuntimeError('temporal/query-override probe is eval-only; disable it before training')

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Temporal query propagation probe: append previous-frame queries at the configured decoder layer, or
        # fuse their previous-frame results after the decoder. Disabled by default; skipped during training.
        use_temporal = self.temporal_enabled and not self.training
        fuse_after = use_temporal and self.temporal_fuse_after
        collect_states = use_temporal or self.return_all_layers
        self._last_n_current = embed.shape[1]
        self._last_layer_states = None
        self._last_layer_widths = None

        inject_at = None
        n_temporal = 0
        if use_temporal and not fuse_after:
            gathered = self._gather_temporal_state()
            if gathered is not None:
                embed_add, refer_add = gathered
                n_temporal = embed_add.shape[1]
                inject_at = {self.temporal_injection_layer: (embed_add, refer_add)}
        self._last_decoder_query_count = embed.shape[1] + n_temporal

        # Staged query exit (QD-P): online policies decide inside the decoder; random / gt_oracle use the
        # per-frame masks the caller placed in _exit_masks. Exits and temporal injection are mutually exclusive.
        use_exit = self.exit_schedule is not None and not self.training
        if use_exit and use_temporal:
            raise RuntimeError('query-exit probe cannot be combined with the temporal probe')
        exit_plan = None
        exit_selector = None
        if use_exit:
            if self.exit_policy in ('random', 'gt_oracle'):
                exit_plan = self._exit_masks
            else:
                exit_selector = self._select_exit_mask
        exit_ann = None

        # Decoder
        t0 = t1 = None
        if self.time_decoder_cuda and torch.cuda.is_available() and not self.training:
            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0.record()
        if use_exit:
            dec_bboxes, dec_scores, exit_ann = self.decoder(
                embed,
                refer_bbox,
                feats,
                shapes,
                self.dec_bbox_head,
                self.dec_score_head,
                self.query_pos_head,
                attn_mask=attn_mask,
                exit_plan=exit_plan,
                exit_selector=exit_selector,
            )
            layer_states = None
        elif collect_states:
            dec_bboxes, dec_scores, _, layer_states = self.decoder(
                embed,
                refer_bbox,
                feats,
                shapes,
                self.dec_bbox_head,
                self.dec_score_head,
                self.query_pos_head,
                attn_mask=attn_mask,
                inject_at=inject_at,
                return_layer_states=True,
            )
        else:
            dec_bboxes, dec_scores = self.decoder(
                embed,
                refer_bbox,
                feats,
                shapes,
                self.dec_bbox_head,
                self.dec_score_head,
                self.query_pos_head,
                attn_mask=attn_mask,
            )
            layer_states = None
        if t0 is not None:
            t1.record()
            torch.cuda.synchronize()
            self._last_decoder_ms = t0.elapsed_time(t1)

        if self.training and dn_meta is None:
            # Touch denoising_class_embed so DDP sees it as used when batch has zero GTs.
            dec_bboxes = dec_bboxes + 0 * self.denoising_class_embed.weight.sum()
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x

        # Source annotation for the final top-k: current encoder queries come first, propagated queries last.
        # injection_layer = -2 (never injected), 0/1/2 (layer before which the query was appended), -1 (fused
        # after the decoder, T3O).
        query_source = None
        query_source_depth = None
        query_injection = None
        if use_temporal and self._temporal_state is not None:
            n_cur = self._last_n_current
            st = self._temporal_state
            m_t = st['idx'].shape[1]
            bsq = dec_bboxes.shape[1]  # batch dim of the stacked decoder output (1, bs, q, ...)
            if fuse_after:
                # Lower-bound reference: previous-frame L3 box + logits re-enter only the top-k, no decoder pass.
                t_box = st['boxes'][3]
                if self.temporal_motion_alpha:
                    t_box = t_box + self.temporal_motion_alpha * st['velocity']
                    t_box = t_box.clamp(1e-4, 1.0 - 1e-4)
                dec_bboxes = torch.cat([dec_bboxes, t_box.unsqueeze(1)], 2)
                dec_scores = torch.cat([dec_scores, st['logits3'].unsqueeze(1)], 2)
                inj_ann = -1
            else:
                inj_ann = self.temporal_injection_layer
            query_source = torch.cat(
                [
                    torch.zeros(bsq, n_cur, dtype=torch.long, device=dec_bboxes.device),
                    torch.ones(bsq, m_t, dtype=torch.long, device=dec_bboxes.device),
                ],
                1,
            )
            query_source_depth = torch.cat(
                [
                    torch.zeros(bsq, n_cur, dtype=torch.long, device=dec_bboxes.device),
                    torch.full((bsq, m_t), self.temporal_source_depth, dtype=torch.long, device=dec_bboxes.device),
                ],
                1,
            )
            query_injection = torch.cat(
                [
                    torch.full((bsq, n_cur), -2, dtype=torch.long, device=dec_bboxes.device),
                    torch.full((bsq, m_t), inj_ann, dtype=torch.long, device=dec_bboxes.device),
                ],
                1,
            )
        # (bs, num_queries, 4), (bs, num_queries, nc)
        y = self.postprocess(
            dec_bboxes.squeeze(0),
            dec_scores.squeeze(0).sigmoid(),
            query_source=query_source,
            query_source_depth=query_source_depth,
            query_injection=query_injection,
            query_exit_layer=exit_ann,
        )
        if use_temporal:
            self._update_temporal_state(layer_states, embed, enc_bboxes)
        if collect_states:
            self._last_layer_states = {k: tuple(t.detach() for t in v) for k, v in layer_states.items()}
            self._last_layer_widths = {k: v[0].shape[1] for k, v in layer_states.items()}
        if use_exit:
            self._last_exit_ann = exit_ann.detach()
            self._last_dec_boxes = dec_bboxes.detach()  # merged per-query boxes of the exit path (diagnostics)
            ann = exit_ann[0]
            q0 = ann.shape[0]
            widths = [q0]
            for li in range(1, self.decoder.eval_idx + 1):
                widths.append(int(((ann == -1) | (ann >= li)).sum()))
            self._last_exit_widths = widths
        if t0m is not None:
            t1m.record()
            torch.cuda.synchronize()
            self._last_model_ms = t0m.elapsed_time(t1m)
        return y if self.export else (y, x)

    # ------------------------------------------------------------------ B1a pair training API
    @torch.no_grad()
    def extract_pair_state(self, x: list[torch.Tensor]) -> dict:
        """Extract the previous frame's temporal state from FPN features (B1a training).

        Uses the same selection rule as deployment: top-M queries by last-layer max-class sigmoid score
        (deterministic torch.topk). Must be called in eval mode so batchnorm uses running statistics,
        matching how the state is produced at inference time.

        Args:
            x (list[torch.Tensor]): FPN feature maps of the previous frame.

        Returns:
            (dict): hidden/boxes keyed by source depth (2 = after L2, 3 = after L3), plus selection scores
                and query indices. All tensors are detached clones; contains no GT and no current-frame data.
        """
        if self.training:
            raise RuntimeError('extract_pair_state must run in eval mode (caller switches to eval first)')
        spec = self.b1a_temporal_spec
        feats, shapes = self._get_encoder_input(x)
        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes)
        _, _, _, layer_states = self.decoder(
            embed, refer_bbox, feats, shapes, self.dec_bbox_head, self.dec_score_head, self.query_pos_head,
            return_layer_states=True)
        li = self.decoder.eval_idx  # S3 = state after the eval layer; S2 = one layer earlier
        h3, b3, l3, _ = layer_states[li]
        h2, b2, _, _ = layer_states[li - 1]
        scores = l3.sigmoid().amax(-1)  # (bs, n)
        m = min(int(spec['M']), h3.shape[1])
        top = scores.topk(m, dim=1).indices
        bi = torch.arange(embed.shape[0], device=embed.device).unsqueeze(-1)

        def pick(t: torch.Tensor) -> torch.Tensor:
            return t[bi, top].detach().clone()

        return {
            'hidden': {2: pick(h2), 3: pick(h3)},
            'boxes': {2: pick(b2), 3: pick(b3)},
            'scores': pick(scores),
            'idx': top.detach().clone(),
        }

    def forward_train_pair(self, x: list[torch.Tensor], batch: dict, state: dict | None) -> tuple:
        """Run the B1a training forward of the current frame with temporal query injection.

        The current-frame path is pristine. The (detached) temporal state passes through the trainable
        source adapter and embedding, so from that branch gradients reach only the adapter/embedding.
        Injected reference boxes stay in sigmoid space, matching inject_at semantics.

        Args:
            x (list[torch.Tensor]): FPN feature maps of the current frame (grad enabled).
            batch (dict): GT targets of the current frame.
            state (dict | None): extract_pair_state output for the previous frame; None disables injection
                (M=0 degeneration, outputs collapse to the control path).

        Returns:
            (tuple): (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta) where dec_* are per-layer
                lists when injection makes per-layer query counts differ (stacked tensor otherwise); the
                model loss prepends the encoder output as the first element.
        """
        from ultralytics.models.utils.ops import get_cdn_group

        spec = self.b1a_temporal_spec
        feats, shapes = self._get_encoder_input(x)
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch, self.nc, self.num_queries, self.denoising_class_embed.weight, self.num_denoising,
            self.label_noise_ratio, self.box_noise_scale, self.training)
        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        inject_at = None
        if state is not None:
            sd = int(spec['source_depth'])
            li = int(spec['injection_layer'])
            embed_add = self.temporal_adapter(state['hidden'][sd]) + self.temporal_src_embed
            refer_add = state['boxes'][sd].clamp(1e-4, 1.0 - 1e-4)
            inject_at = {li: (embed_add, refer_add)}

        dec_bboxes, dec_scores = self.decoder(
            embed, refer_bbox, feats, shapes, self.dec_bbox_head, self.dec_score_head, self.query_pos_head,
            attn_mask=attn_mask, inject_at=inject_at)
        # NOTE: no DDP zero-touch of denoising_class_embed here (B1a is single-GPU; dec_* may be lists).
        return dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta

    def postprocess(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        query_source: torch.Tensor | None = None,
        query_source_depth: torch.Tensor | None = None,
        query_injection: torch.Tensor | None = None,
        query_exit_layer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Post-process predictions to select top-k detections.

        Args:
            boxes (torch.Tensor): Predicted bounding boxes with shape (batch_size, num_queries, 4) in xywh format.
            scores (torch.Tensor): Class scores with shape (batch_size, num_queries, nc).
            query_source (torch.Tensor, optional): Per-query origin flags with shape (batch_size, num_queries),
                0 for current-frame encoder queries and 1 for propagated temporal queries. Only used to annotate
                the final top-k outputs in `self._last_topk_meta`; never changes the outputs themselves.
            query_source_depth (torch.Tensor, optional): Per-query source-depth annotation (0..3).
            query_injection (torch.Tensor, optional): Per-query injection-layer annotation
                (-2 never injected, 0/1/2 injected before that layer, -1 fused after the decoder).
            query_exit_layer (torch.Tensor, optional): Per-query exit annotation for the QD-P probe
                (-1 ran to the eval layer, i exited after layer i). Defaults to -1 everywhere.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, num_queries, 6) and last dimension format [cx,
                cy, w, h, max_class_prob, class_index].
        """
        topk = self._output_topk_override or self.b1a_eval_topk or self.num_queries
        scores, index = scores.flatten(1).topk(topk)
        # CoreML MIL lacks integer floor-div and mod lowering: use torch.div(rounding_mode="floor") and (index - q*nc).
        query_idx = torch.div(index, self.nc, rounding_mode="floor")
        boxes = boxes.gather(dim=1, index=query_idx.unsqueeze(-1).expand(-1, -1, 4).long())
        src = (
            torch.zeros_like(query_idx)
            if query_source is None
            else torch.gather(query_source, 1, query_idx.long())
        )
        src_depth = (
            torch.zeros_like(query_idx)
            if query_source_depth is None
            else torch.gather(query_source_depth, 1, query_idx.long())
        )
        inj_layer = (
            torch.full_like(query_idx, -2)
            if query_injection is None
            else torch.gather(query_injection, 1, query_idx.long())
        )
        exit_layer = (
            torch.full_like(query_idx, -1)
            if query_exit_layer is None
            else torch.gather(query_exit_layer, 1, query_idx.long())
        )
        self._last_topk_meta = {
            'query_idx': query_idx.detach(),
            'scores': scores.detach(),
            'cls': (index - query_idx * self.nc).detach(),
            'source': src.detach(),
            'source_depth': src_depth.detach(),
            'injection_layer': inj_layer.detach(),
            'exit_layer': exit_layer.detach(),
        }
        return torch.cat([boxes, scores[..., None], (index - query_idx * self.nc)[..., None].float()], dim=-1)

    @staticmethod
    def _generate_anchors(
        shapes: list[list[int]],
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        eps: float = 1e-2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate anchor bounding boxes for given shapes with specific grid size and validate them.

        Args:
            shapes (list): List of feature map shapes.
            grid_size (float, optional): Base size of grid cells.
            dtype (torch.dtype, optional): Data type for tensors.
            device (str, optional): Device to create tensors on.
            eps (float, optional): Small value for numerical stability.

        Returns:
            anchors (torch.Tensor): Generated anchor boxes.
            valid_mask (torch.Tensor): Valid mask for anchors.
        """
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, list[list[int]]]:
        """Process and return encoder inputs by getting projection features from input and concatenating them.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.

        Returns:
            feats (torch.Tensor): Processed features.
            shapes (list): List of feature map shapes.
        """
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(
        self,
        feats: torch.Tensor,
        shapes: list[list[int]],
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate and prepare the input required for the decoder from the provided features and shapes.

        Args:
            feats (torch.Tensor): Processed features from encoder.
            shapes (list): List of feature map shapes.
            dn_embed (torch.Tensor, optional): Denoising embeddings.
            dn_bbox (torch.Tensor, optional): Denoising bounding boxes.

        Returns:
            embeddings (torch.Tensor): Query embeddings for decoder.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            enc_bboxes (torch.Tensor): Encoded bounding boxes.
            enc_scores (torch.Tensor): Encoded scores.
        """
        bs = feats.shape[0]
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
            self.shapes = shapes

        # Query-count concepts are split for the eval-only probe: the encoder top-k count can differ from
        # self.num_queries (which still drives postprocess) so a control group can run 375 current queries.
        nq = self._encoder_query_override or self.num_queries

        # Prepare input for decoder
        features = self.enc_output(self.valid_mask * feats)  # bs, h*w, 256
        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs*nq,)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, nq, dim=1).indices.view(-1)
        # (bs*nq,)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, nq).view(-1)

        # (bs, nq, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, nq, -1)
        # (bs, nq, 4)
        top_k_anchors = self.anchors[:, topk_ind].view(bs, nq, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, nq, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def _reset_parameters(self):
        """Initialize or reset the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    """v10 Detection head from https://arxiv.org/pdf/2405.14458.

    This class implements the YOLOv10 detection head with dual-assignment training and consistent dual predictions for
    improved efficiency and performance.

    Attributes:
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum number of detections.
        cv3 (nn.ModuleList): Light classification head layers.
        one2one_cv3 (nn.ModuleList): One-to-one classification head layers.

    Methods:
        __init__: Initialize the v10Detect object with specified number of classes and input channels.
        forward: Perform forward pass of the v10Detect module.
        bias_init: Initialize biases of the Detect module.
        fuse: Remove the one2many head for inference optimization.

    Examples:
        Create a v10Detect head
        >>> v10_detect = v10Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = v10_detect(x)
    """

    end2end = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Initialize the v10Detect object with the specified number of classes and input channels.

        Args:
            nc (int): Number of classes.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, end2end=True, ch=ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    def fuse(self):
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = None


class SemanticSegment(nn.Module):
    """YOLO semantic segmentation head for per-pixel classification.

    This head produces dense per-pixel class predictions. Unlike instance segmentation, no bounding boxes or instance
    masks are produced.

    Attributes:
        nc (int): Number of semantic classes.
        nl (int): Number of input feature levels.
        stride (torch.Tensor): Feature map strides.
        export (bool): Export mode flag.
        format (str): Export format.
        classifier (nn.Sequential): Final convolutional classifier head.
        aux_head (nn.Sequential | None): Auxiliary classifier on P4 for deep supervision.
    """

    export = False  # export mode
    format = None  # export format
    bake_argmax = False  # export: emit a baked [B, H, W] class map (set by the exporter for TensorRT>=10)

    def __init__(self, nc=19, ch=()):
        """Initialize the semantic segmentation head.

        Args:
            nc (int): Number of semantic classes.
            ch (tuple): Tuple of channel sizes from neck feature maps (P3, P4).
        """
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.stride = torch.zeros(self.nl)

        c_mid = ch[0]  # use P3 channel width as intermediate dimension
        # Final classifier
        self.classifier = nn.Sequential(Conv(c_mid, c_mid, 3), nn.Conv2d(c_mid, nc, 1))
        # Auxiliary head on P4 (index 1) for training
        self.aux_head = nn.Sequential(Conv(ch[1], c_mid, 3), nn.Conv2d(c_mid, nc, 1)) if len(ch) > 1 else None

    def forward(self, x):
        """Forward pass: fuse multi-scale features and predict per-pixel classes.

        Args:
            x (list[torch.Tensor]): List of feature maps [P3, P4].

        Returns:
            (torch.Tensor | tuple): Logits of shape [B, nc, H/8, W/8] during training (or a (main, aux) tuple when
                aux_head is present) and inference. ONNX, MNN, and TensorRT>=10 export bake in the argmax and return a
                compact class map of shape [B, H, W] (uint8 when nc <= 256, else int32). Other export formats return
                upsampled logits of shape [B, nc, H, W].
        """
        # Classify
        logits = self.classifier(x[0])  # [B, nc, H/8, W/8]
        if self.training:
            if self.aux_head is not None:
                return logits, self.aux_head(x[1])  # main + aux (P4)
            return logits
        if self.export:
            y = F.interpolate(logits, scale_factor=8, mode="bilinear", align_corners=False)  # [B, nc, H, W]
            # Bake argmax: emit [B, H, W] class map, shrinking the D2H copy ~80x. ONNX/MNN preserve the
            # integer output; TensorRT only supports uint8 graph outputs on TRT>=10, so engine is gated by the exporter.
            if self.format in {"onnx", "mnn"} or (self.format == "engine" and self.bake_argmax):
                cls = y.argmax(1) if self.nc > 1 else y.squeeze(1) > 0
                return cls.to(torch.uint8 if self.nc <= 256 else torch.int32)
            return y
        return logits
