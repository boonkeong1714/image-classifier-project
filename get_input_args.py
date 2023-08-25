import argparse

def get_input_args():
    parser = argparse.ArgumentParser(prog='Project 2', description="Trains a new network on a dataset of images")
    parser.add_argument('--arch', dest='arch', default='vgg16', choices = ["vgg16", "alexnet"])
    parser.add_argument('--epochs', dest='epochs', default=4, type=int)
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.0013, type=float)
    parser.add_argument('--hidden_units', dest='hidden_units', default='512', type=int)
    parser.add_argument('--gpu', dest='gpu', default='gpu')
    parser.add_argument('--source_dir', dest='source_dir', default='flowers')
    parser.add_argument('--save', dest="checkpoint_path", default="checkpoint3.pth")
    parser.add_argument('--predict', dest='image_path', default='flowers/test/5/image_05159.jpg', help='Path of image to test')
    
    return parser.parse_args()
