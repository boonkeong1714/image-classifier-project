import torch
from torchvision import models
import numpy as np
from PIL import Image
import json
from get_input_args import get_input_args

import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

def process_image(image):
    width, height = image.size
    aspect_ratio = width / height

    # Resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    if width < height:
        new_width = 256
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = 256
        new_width = int(new_height * aspect_ratio)

    resized_image = image.resize((new_width, new_height))

    # Crop out the center 224x224 portion of the image
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = (new_width + 224) / 2
    bottom = (new_height + 224) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))

    # Convert the image to Numpy array
    np_image = np.array(cropped_image)

    # Normalize the image
    np_image = np_image / 255.0
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stds

    output_image = np_image.transpose((2, 0, 1))
    
    return output_image


def predict(image_path, checkpoint_path, models_arch, topk=5):
    model, class_to_idx = load_checkpoint(checkpoint_path, models_arch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("---device---", device)
    
    with Image.open(image_path) as image: 
        process_image_data = process_image(image)
    img = torch.from_numpy(process_image_data)
    img = img.to(device)
    img_reshape = img.unsqueeze(0).float()

    with torch.no_grad():
        model.eval()
        model.to(device)
        
        logps = model(img_reshape)
        ps = torch.exp(logps)

        probs, labels = ps.topk(topk, dim=1)
        rounded_probs = [round(num, 4) for num in probs.tolist()[0]]

        class_to_idx_convert = {class_to_idx[i]: i for i in class_to_idx}

        classes = []
        for label in labels.cpu().numpy()[0]:
            classes.append(class_to_idx_convert[label])
        
        return rounded_probs, classes


def load_checkpoint(checkpoint_path, models_arch):
    checkpoint = torch.load(checkpoint_path)

    if models_arch == 'vgg16':
        model = models.vgg16(weights=True)
    elif models_arch == 'alexnet':
        model = models.alexnet(weights=True)

    # epochs = checkpoint['epochs']
    # learning_rate = checkpoint['learning_rate']
    # model = checkpoint['model']
    model.classifier = checkpoint['model_classifier']
    model.load_state_dict = checkpoint['model_state_dict']
    model.optimizer = checkpoint['optimizer_state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model, model.class_to_idx


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    
    return ax


def main(): 
    input_args = get_input_args()

    models_arch = input_args.arch
    checkpoint_path = input_args.checkpoint_path
    img_path = input_args.image_path

    print('----- The Input Arguments -----')
    print('pre-trained model --arch:', models_arch)
    print('checkpoint_path --save:', checkpoint_path)
    print('image_path --predict:', img_path)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    predict_probs, predict_classes = predict(img_path, checkpoint_path, models_arch)

    print('----- The Prediction Results -----')
    print('File selected: ' + img_path)
    print('The top 5 predicted flower classes are:', predict_classes)
    print('and their probablities are', predict_probs)

    correct__image_class = img_path.split("/")[2]

    classes = []
    for predict__class in predict_classes:
        classes.append(cat_to_name[predict__class])

    # Plotting image with correct answer
    fig = plt.figure(figsize = (8, 5))
    ax = plt.subplot(2,1,1)
    ax.set_title("this flower should be: " + cat_to_name[correct__image_class])
    plt.axis('off')
    with Image.open(img_path) as image: 
        imshow(process_image(image), ax, title="lol");

    # Plotting probs in bar chart
    plt.subplot(2,1,2)
    sns.barplot(x=predict_probs, y=classes, color="royalblue", width=0.5);
    plt.show()

if __name__ == '__main__':
    main()