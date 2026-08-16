# Model Checkpoints & Weights

This directory holds trained NAFNet-SR model checkpoints.

## Checkpoint File
- **`nafnet_sr_best.pt`**: Best model checkpoint (PyTorch state dict).

## Generating / Training Weights
To train the model from scratch and generate the best checkpoint:
```bash
python train.py --full_train
```
The checkpoint with the highest validation PSNR will automatically be saved to `weights/nafnet_sr_best.pt`.

## Checkpoint Usage in Inference
The inference pipeline automatically loads `weights/nafnet_sr_best.pt` by default:
```bash
python inference.py --input_dir /path/to/test/NoisyLR --output_dir /path/to/predictions
```
