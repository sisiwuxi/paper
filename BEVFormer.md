# reference
- https://arxiv.org/abs/2203.17270
- https://github.com/zhiqi-li/BEVFormer

# camera-only 3D perception
- mlti-camera inputs, perceive 
- dataset
  - nuScenes: #6
  - Waymo: #5,270
# BEV: bird's-eye-view
## pros
- fuse multi-camera features in early stage
- straight forward to combine with other modalities
- readily consumable by downstream components such as prediction and planning
## BEV = 2D
- 2d backbone
- features
- task head

- 2d backbone
- BEV encoder
- task head

# view transformation
- camera projection
- from 3D to 2D
  - issue: multiple 3D points will hit the same 2D pixel
    - these 3D points with different depth
- from 2D to 3D
  - issue: depth is unknown
- no matter what, the transformation is ill-posed

# lift-splat-shoot: 2D->3D
- using categorical distribution over depth instead of depth estimates
- strength
  - generate representation as all possible and sparse
- weakness
  - the generated BEV is discontinuous and sparse
  - the fusion process is inefficient

# OFT: 3D->2D
- obtain image features that corresponds to predifined 3D anchors
- strength
  - dense or sparse BEV features maps
  - efficient compared to 2D to 3D
- weakness
  - false positive BEV features

# caveat 1
- because of the geometry of road, projection cannot precisely project corresponding point to BEV
- if some part is occluded, the projection will be wrong
- need to find the relation fron BEV to camera

# dense attn -> sparse attn
- dense attention  needs to consider all
- sparse attention only focus on the ROIs calculated by the canera parameters(intrinsic & extrinsic), which is effective

# side issue in caveat 1
- multi-scale feature maps
- memoru-efficient
- computation-efficient

# BEV former
- what's in here at timestamp?
- lookup & aggregate the spatial information
  - multi-camera images at timestamp t
- temporal attention

# overall architecture
- history BEV B_{t-1} + BEV queries Q
  - Additional consumption < 1% latency
- temporal self-attention
- add & norm
- spatial cross-attention
- add & norm
- feed forward
- add & norm
- current BEV

# spatial cross-attention
- space
- key-steps
  - lift each BEV query to be a pillar
  - project the 3D points in pillar to 2D points in views
  - sample features from ROIs in hit views
  - fuse by weight
- ex
  - x=200,y=200
  - sampling z= -1,1,2,3,...
  - 3D->2D
- standardize eagle-car

# temporal information
- timing
- key-steps
  - align two BEV maps according to the ego motion
  - sample features from both past and the current
  - weighted summation of sampled features from past and the current BEV maps
  - use RNN-style to literately colllevt history BEV features

# psudo-2D features
- map segmentation
- 3D object detection
- with the shared BEV fdeatures, the results of the map segmentation and 3D object detection is consistent

# 3Dfy DETR detector
- do not need post process
- set of box predictions
- DD3D,detr3d

# EXP3: temporal clues matters
- higher recall, especially for low-visible objects
- more accurate location estimation
- very accurate estimate of velocity

# multimodality fusion
- sensor configutations
- BEVFusion
  - multi view RGB image -> BEV
  - LiDAR -> BEV
    - LiDAR provide deep information
    - cost
- depth
  - list-splat-shoot
  - CADDN
  - BEVDepth
  - LSS: replacing estimated depth values with depth distributions
- polar representation
  - cartesian & polar coordinates

# end-to-end
- detection
- planning: transformer