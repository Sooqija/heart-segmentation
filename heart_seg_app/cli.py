import argparse
from heart_seg_app.train import train

def main():
    parser = argparse.ArgumentParser(description="Heart Segmentation App")
    parser.add_argument("mode", choices=["train", "eval"], help="Mode: train or eval")
    parser.add_argument('--model', type=str, choices=["unet3d", "unetr"], help="Model type", required=True)
    parser.add_argument('--checkpoint', type=str, help="Path to model weights", default=None)
    args = parser.parse_args()

    if args.mode == "train":
        print("Training mode")
        if args.checkpoint:
            print(f"Using checkpoint: {args.checkpoint}")
            train(model=args.model, checkpoint=args.checkpoint)
    elif args.mode == "eval":
        print("Evaluation mode")

if __name__ == "__main__":
    main()
