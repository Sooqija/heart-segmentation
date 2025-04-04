import argparse
from heart_seg_app.train import train
from heart_seg_app.evaluate import evaluate

def main():
    parser = argparse.ArgumentParser(description="Heart Segmentation App")
    parser.add_argument("mode", choices=["train", "eval"], help="Mode: train or eval")
    
    parser.add_argument('-img', '--image_dir', type=str, help="Path to image data", required=True)
    parser.add_argument('-lbl', '--label_dir', type=str, help="Path to label data")
    parser.add_argument('--dataset_config', type=str, help="Path to dataset config json file")
    parser.add_argument('--split_ratios', type=float, nargs=3, help="Train, validation, and test split ratios", default=(0.75, 0.2, 0.05))
    parser.add_argument('--seed', type=int, help="Random seed for reproducibility")
    
    parser.add_argument('--model', type=str, choices=["unet3d", "unetr"], help="Model type", required=True)
    parser.add_argument('--checkpoint', type=str, help="Path to model weights", default=None)
    
    parser.add_argument('--output_dir', type=str, help="Directory to save model checkpoints and training statistics.")
    parser.add_argument('-t', '--tag', type=str, help="Tag to label and track the current training run.", required=True)
    
    parser.add_argument('--epochs', type=int, help="Numbers of epochs", default=250)
    args = parser.parse_args()

    if args.mode == "train":
        print("Training mode")
        train(model=args.model,
              image_dir=args.image_dir,
              label_dir=args.label_dir,
              dataset_config=args.dataset_config,
              split_ratios=(args.split_ratios),
              seed=args.seed,
              checkpoint=args.checkpoint,
              output_dir=args.output_dir,
              tag=args.tag,
              epochs=args.epochs)
    elif args.mode == "eval":
        print("Evaluation mode")
        
        if args.checkpoint is None:
            raise ValueError("Checkpoint is required for evaluation mode.")
        if args.dataset_config is None:
            raise ValueError("Dataset config is required for evaluation mode.")
        
        evaluate(model=args.model,
                 image_dir=args.image_dir,
                 label_dir=args.label_dir,
                 dataset_config=args.dataset_config,
                 checkpoint=args.checkpoint,
                 output_dir=args.output_dir,
                 tag=args.tag)
        

if __name__ == "__main__":
    main()
