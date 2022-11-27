import argparse
import sys
import torch

parser = argparse.ArgumentParser(description="evaluate all models in one shot")
parser.add_argument('-epochs', help='number of epochs for which model has been trained', type=int, default=20)
parser.add_argument('-first_time', help='models from epoch 7 should be treat differently', type=bool, default=False)

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

def evaluate():
    for e in range(1, args.epochs+1):
        if args.first_time and e < 7:
            model = torch.load(f'./output/model_{e}.pt')
        else:
            model = torch.load(f'./output/model_{e}.pt')['model']
    