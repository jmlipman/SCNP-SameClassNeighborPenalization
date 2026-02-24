# SCNP Experiments

We evaluated SCNP combined with Cross Entropy and Dice loss against several topology and non-topology loss functions on 13 datasets, including medical, non-medical datasets, datasets with tubular and non-tubular structures, single- and multi-class datasets, and on semantic and instance segmentation.
You can find a summary of the datasets in the Appendix.

We run our experiments on three frameworks: nnUNetv2, Detectron2 with DeepLabv3+, and InstanSeg.
Here, we provide the files that we modified to make SCNP run on these frameworks.
In order to reproduce our results, 1) download these frameworks from their official repositories, 2) make sure they can run on the datasets, and 3) modify the files involved in the optimization.
Below, we provide a summary with the files that we modified and their location.

Note: These are the files used to run SCNP(CEDice) (i.e., Cross Entropy and Dice loss optimizing only the SCNP-normalized logits).

## nnUNetv2

Official repository: https://github.com/MIC-DKFZ/nnUNet/tree/master

| File | Path |
| -------- | ------- |
| [nnUNetTrainerSCNPLoss.py](nnUNetv2/nnUNetTrainerSCNPLoss.py) | nnunetv2/training/nnUNetTrainer/nnUNetTrainerSCNPLoss.py |
| [compound_losses_scnp.py](nnUNetv2/compound_losses_scnp.py) | nnunetv2/training/loss/compound_losses_scnp.py |

## Detectron with Deeplabv3+

Official repository: https://github.com/facebookresearch/detectron2

| File | Path |
| -------- | ------- |
| [semantic_seg.py](Detectron2/semantic_seg.py) | detectron2/projects/DeepLab/deeplab/semantic_seg.py |
| [loss.py](Detectron2/loss.py) | detectron2/projects/DeepLab/deeplab/loss.py |

## InstanSeg

Official repository: https://github.com/instanseg/instanseg


| File | Path |
| -------- | ------- |
| [compound_losses_scnp.py](InstanSeg/compount_losses_scnp.py) | instanseg/utils/loss/compound_losses_scnp.py |
| [instanseg_loss.py](InstanSeg/instanseg_loss.py) | instanseg/utils/loss/instanseg_loss.py |
