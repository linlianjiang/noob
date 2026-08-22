from mmdet3d.registry import MODELS

from .frnet_head import FRHead


@MODELS.register_module()
class ObsFRHead(FRHead):
    """FRHead that also stashes ``f_r`` (the classifier input) in
    ``voxel_dict`` for Sec. III-C. No new parameters, so pretrained FRNet
    weights load unchanged.
    """

    def forward(self, voxel_dict: dict) -> dict:
        point_feats_backbone = voxel_dict['point_feats_backbone'][0]
        point_feats = voxel_dict['point_feats'][:-1]
        voxel_feats = voxel_dict['voxel_feats'][0]
        voxel_feats = voxel_feats.permute(0, 2, 3, 1)
        pts_coors = voxel_dict['coors']
        map_point_feats = voxel_feats[pts_coors[:, 0], pts_coors[:, 1],
                                      pts_coors[:, 2]]

        for i, mlp in enumerate(self.mlps):
            map_point_feats = mlp(map_point_feats)
            if i == 0:
                map_point_feats = map_point_feats + point_feats_backbone
            else:
                map_point_feats = map_point_feats + point_feats[-i]

        voxel_dict['point_feats_head'] = map_point_feats
        voxel_dict['seg_logit'] = self.cls_seg(map_point_feats)
        return voxel_dict
