# Enhancing 2D hand pose estimation and tracking on surgical videos using attention mechanism

This source code was mainly copied from the [official repository](https://https://github.com/MichiganCOG/Surgical_Hands_RELEASE) of [Temporally Guided Articulated Hand Pose Tracking in Surgical Videos](https://arxiv.org/abs/2101.04281)

## Datasets
 - **Mixed Hands** Dataset used for image pretraining are the Manual and Synthetic hand datasets (hence Mixed Hands) from the [CMU Panoptic Dataset](http://domedb.perception.cs.cmu.edu/handdb.html)
	- Extract to `$ROOT/data` directory and configure using `scripts/gen_json_mixed_hands.py`
 - **Surgical Hands** Our newly collected dataset that contains videos of surgical procedures accompanied with bounding box, pose, and tracking annotations. 
      - Download the following and extract to `$ROOT/data` directory
          - [Surgical Hands dataset](https://drive.google.com/file/d/1l5_4rlZLvOim34uHCKic4GUXvXfjDN_9/view?usp=sharing)
          - [Hand Detections](https://drive.google.com/file/d/1dWhZF595ixS-XBIeawaS3mY01yfsE_BO/view?usp=sharing)
      - Configure using `scripts/gen_json_surgical_hands_folds_n_train_val_test.py --source_json_file annotations.json --source_res_dir images_dir --target_json_dir splitted_json_files --target_poseval_dir poseval_dir
      ` (for ground truth)
      - Unlike the experiments conducted in [1], we evaluate the models on our own split. We choose the 4 best and 4 worst performing fold evaluated using the pretrained weights provided in [1]. Then, we combine 2 videos from the 4 best and 2 videos from the 4 worst to create validation set and test set respectively.

## Weights
- Download and extract to `$ROOT/weights` directory
    - ResNet152 ImageNet [weights](https://drive.google.com/file/d/14u4TYEpu6d6Eh4PsIOjeTYiMfc4nXAMe/view?usp=sharing)
    - Pretrained [weights](https://drive.google.com/drive/folders/1upSSUr4c2_SMmpzfQumoevNYhpig0UuW?usp=sharing) on Mixed Hands image dataset
    - Baseline and our models [weights](https://drive.google.com/drive/folders/1CAyzU6bAeiLxND7KpF6lBIcURuH9Jvd6?usp=sharing) (trained on Surgical Hands)
     
## Training and Evaluation
### Pre-train on larger image dataset
`python train.py --cfg_file ./cfgs/config_hand_resnet.yaml --json_path ./data/hand_labels_mixed --model baseline --epoch 75 --lr 1e-4 --batch_size 16 --milestones 40,60`

### Finetune on our (Surgical Hands) dataset
- (Baseline) `python train.py --cfg_file ./cfgs/config_train_surgical_hands_baseline.yaml --json_path ./data/pub_surgical/annotations_fold99 --pretrained ./weights/Mixed_Hands/Mixed_Hands_best_model.pkl --tags folda99`

- (Our models) `python train.py --cfg_file ./cfgs/config_train_surgical_hands.yaml --json_path ./data/pub_surgical/annotations_fold99 --pretrained ./weights/Mixed_Hands/Mixed_Hands_best_model.pkl --tags folda99 --model (see $ROOT/models)`

 
### Evaluation
For evaluation, we modify the [Poseval Evaluation repository](https://github.com/leonid-pishchulin/poseval) for hands instead of human pose (amongst other threshold and validation changes). All code is contained within [poseval\_hand](https://github.com/MichiganCOG/Surgical_Hands_RELEASE/tree/main/poseval_hand).

- (Baseline) `python eval.py --cfg_file ./cfgs/config_eval_surgical_hands_baseline.yaml --json_path ./data/pub_surgical/annotations_fold99 --pretrained ./weights/(your model weights) --tags folda99`

- (Our models) `python eval.py --cfg_file ./cfgs/config_eval_surgical_hands.yaml --json_path ./data/pub_surgical/annotations_fold99 --pretrained ./weights/(your model weights) --tags folda99 --model (see $ROOT/models)`

### Visualization
- (Baseline) `python eval_cycle.py --cfg_file cfgs/config_eval_surgical_hands_baseline.yaml --json_path ./data/pub_surgical/annotations_folda99 --tags folda99 --pretrained ./weights/(your model weights) --acc_metric Save_Video_Keypoints`

- (Our models) `python eval_cycle.py --cfg_file cfgs/config_eval_surgical_hands.yaml --json_path ./data/pub_surgical/annotations_folda99 --tags folda99 --pretrained ./weights/(your model weights) --acc_metric Save_Video_Keypoints`


### References
[1] Louis, N., Zhou, L., Yule, S.J., Dias, R.D., Manojlovich, M., Pagani, F.D., Likosky,
D.S., Corso, J.J.: Temporally guided articulated hand pose tracking in surgical
videos (2021)