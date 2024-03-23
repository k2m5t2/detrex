# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers import MLP, GenerateDNQueries, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils.misc import inverse_sigmoid

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances


class DNDETR_ONNX(nn.Module):
    """Implement DN-DETR in `DN-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2201.12329>`_

    Args:
        backbone (nn.Module): Backbone module for feature extraction.
        in_features (List[str]): Selected backbone output features for transformer module.
        in_channels (int): Dimension of the last feature in `in_features`.
        position_embedding (nn.Module): Position encoding layer for generating position embeddings.
        transformer (nn.Module): Transformer module used for further processing features and input queries.
        embed_dim (int): Hidden dimension for transformer module.
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        freeze_anchor_box_centers (bool): If True, freeze the center param ``(x, y)`` for
            the initialized dynamic anchor boxes in format ``(x, y, w, h)``
            and only train ``(w, h)``. Default: True.
        select_box_nums_for_evaluation (int): Select the top-k confidence predicted boxes for inference.
            Default: 300.
        denoising_groups (int): Number of groups for noised ground truths. Default: 5.
        label_noise_prob (float): The probability of the label being noised. Default: 0.2.
        box_noise_scale (float): Scaling factor for box noising. Default: 0.4.
        with_indicator (bool): If True, add indicator in denoising queries part and matching queries part.
            Default: True.
        device (str): Training device. Default: "cuda".
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_features: List[str],
        in_channels: int,
        position_embedding: nn.Module,
        transformer: nn.Module,
        embed_dim: int,
        num_classes: int,
        num_queries: int,
        criterion: nn.Module,
        aux_loss: bool = True,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],
        pixel_std: List[float] = [58.395, 57.120, 57.375],
        freeze_anchor_box_centers: bool = True,
        select_box_nums_for_evaluation: int = 300,
        denoising_groups: int = 5,
        label_noise_prob: float = 0.2,
        box_noise_scale: float = 0.4,
        with_indicator: bool = True,
        device="cuda",
    ):
        super(DNDETR_ONNX, self).__init__()
        # define backbone and position embedding module
        self.backbone = backbone
        self.in_features = in_features
        self.position_embedding = position_embedding

        # project the backbone output feature
        # into the required dim for transformer block
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        # generate denoising label/box queries
        self.denoising_generator = GenerateDNQueries(
            num_queries=num_queries,
            num_classes=num_classes + 1,
            label_embed_dim=embed_dim,
            denoising_groups=denoising_groups,
            label_noise_prob=label_noise_prob,
            box_noise_scale=box_noise_scale,
            with_indicator=with_indicator,
        )
        self.denoising_groups = denoising_groups
        self.label_noise_prob = label_noise_prob
        self.box_noise_scale = box_noise_scale

        # define leanable anchor boxes and transformer module
        self.transformer = transformer
        self.anchor_box_embed = nn.Embedding(num_queries, 4)
        self.num_queries = num_queries

        # whether to freeze the initilized anchor box centers during training
        self.freeze_anchor_box_centers = freeze_anchor_box_centers

        # define classification head and box head
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=4, num_layers=3)
        self.num_classes = num_classes

        # predict offsets to update anchor boxes after each decoder layer
        # with shared box embedding head
        # this is a hack implementation which will be modified in the future
        self.transformer.decoder.bbox_embed = self.bbox_embed

        # where to calculate auxiliary loss in criterion
        self.aux_loss = aux_loss
        self.criterion = criterion

        # normalizer for input raw images
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # The total nums of selected boxes for evaluation
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

        self.init_weights()

    def init_weights(self):
        """Initialize weights for DN-DETR"""
        if self.freeze_anchor_box_centers:
            self.anchor_box_embed.weight.data[:, :2].uniform_(0, 1)
            self.anchor_box_embed.weight.data[:, :2] = inverse_sigmoid(
                self.anchor_box_embed.weight.data[:, :2]
            )
            self.anchor_box_embed.weight.data[:, :2].requires_grad = False

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    def forward(self, batched_inputs):
        """Forward function of `DN-DETR` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        
        images = self.normalizer(batched_inputs)
        
        
        # todo: remove this part, as mask is not needed for batch=1.
        batch_size, _, H, W = images.shape
        img_masks = images.new_zeros(batch_size, H, W)
        
        features = self.backbone(images)[self.in_features[-1]]
        features = self.input_proj(features)
        img_masks = F.interpolate(img_masks[None], size=features.shape[-2:]).to(torch.bool)[0]
        # img_masks = F.interpolate(img_masks[None], scale_factor=(1/32, 1/32)).to(torch.bool)[0]
        pos_embed = self.position_embedding(img_masks)
        
        targets = None
        
        # for vallina dn-detr, label queries in the matching part is encoded as "no object" (the last class)
        # in the label encoder.
        matching_label_query = self.denoising_generator.label_encoder(
            torch.tensor(self.num_classes).to(self.device)
        ).repeat(self.num_queries, 1)
        indicator_for_matching_part = torch.zeros([self.num_queries, 1]).to(self.device)
        matching_label_query = torch.cat(
            [matching_label_query, indicator_for_matching_part], 1
        ).repeat(batch_size, 1, 1) # (num_q, emd-1) + (num_q, 1) -> (num_q, emd) -> (bs, num_q, 1)
        matching_box_query = self.anchor_box_embed.weight.repeat(batch_size, 1, 1) #(bs, num_q, 4)

        if targets is None:
            input_label_query = matching_label_query.transpose(0, 1)  # (num_queries, bs, embed_dim)
            input_box_query = matching_box_query.transpose(0, 1)  # (num_queries, bs, 4)
            attn_mask = None
            denoising_groups = self.denoising_groups
            max_gt_num_per_image = 0
            
        hidden_states, reference_boxes = self.transformer(
            features,
            img_masks,
            input_box_query,
            pos_embed,
            target=input_label_query,
            attn_mask=[attn_mask, None],  # None mask for cross attention
        )

        # Calculate output coordinates and classes.
        reference_boxes = inverse_sigmoid(reference_boxes[-1])  # (bs, num_q, 4)
        anchor_box_offsets = self.bbox_embed(hidden_states[-1]) # (bs, num_q, emd) -> # (bs, num_q, 4)
        outputs_coord = (reference_boxes + anchor_box_offsets).sigmoid()
        outputs_class = self.class_embed(hidden_states[-1])  # (bs, num_q, emd) -> # (bs, num_q, 1)

        # no need for denoising post process, as only matching part remained.
 
        #  return last layer state, so take index=0
        box_cls = outputs_class[0]   # (1, num_q, 1)  ->  (num_q, 1)
        box_pred = outputs_coord[0]   # (1, num_q, 4)  ->  (num_q, 4)
        
        # bs = 1 for onnx converting, so just take the index = 0 for simplification.
        out_cls = box_cls.sigmoid()
        out_box = box_cxcywh_to_xyxy(box_pred)  #  convert to xyxy format, (num_q, 4)
        
        return out_cls, out_box

       ###### The following content is omitted  #######


