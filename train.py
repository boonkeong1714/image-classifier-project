import time, json

import torch
from torch import utils, nn, optim
from torchvision import datasets, transforms, models

from get_input_args import get_input_args


def train(hyperparameters):
    [models_arch, epochs, learning_rate, hidden_units, gpu, source_dir, checkpoint_path] = hyperparameters

    if models_arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        input_nodes = model.classifier[0].in_features
    elif models_arch == 'alexnet':
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        input_nodes = model.classifier[1].in_features

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    output_nodes = len(cat_to_name)

    # Turn off backpropagation (calculating the gradients of tensors)
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
            nn.Linear(input_nodes, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.13),
            nn.Linear(hidden_units, 256),
            nn.ReLU(),
            nn.Dropout(p=0.17),
            nn.Linear(256, output_nodes),
            nn.LogSoftmax(dim=1)
            )
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    if gpu == 'gpu': 
        print('User chooses GPU training')
        if torch.cuda.is_available():
            print('CUDA is available, using GPU')
            device = 'cuda'
        else:
            print('CUDA is unavailable, using CPU instead')
            device = 'cpu'
    else:
        print('User chooses CPU training')
        device = 'cpu'
    model.to(device)

    print('----- Starting training -----')
    start_time = time.time()

    steps = 0
    print_every = 10
    train_losses, valid_losses = [], []
    for epoch in range(epochs):
        print('--- Epoch:', str(epoch), '---')
        train_loss = 0
        for inputs, labels in dataloaders(source_dir, 'train'):
            inputs, labels = inputs.to(device), labels.to(device)
            steps += 1
            print('Steps:', str(steps), end='\r')
            
            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            if steps % print_every == 0:
                valid_loss, valid_accuracy = 0, 0

                # Set the model to evaluation mode where the dropout probability is 0.
                model.eval()

                # Turn off gradients for validation
                with torch.no_grad():
                    for inputs, labels in dataloaders(source_dir, 'valid'):
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        logps = model.forward(inputs)
                        loss = criterion(logps, labels)
                        valid_loss += loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                # Set the model back to train mode
                model.train()

                train_length = len(dataloaders(source_dir, 'train'))
                valid_length = len(dataloaders(source_dir, 'valid'))
                train_losses.append(train_loss/train_length)
                valid_losses.append(valid_loss/valid_length)

                print(f'[Epoch {epoch+1}/{epochs}]',
                  f'Train loss: {train_loss/train_length:.3f} | '
                  f'Valid loss: {valid_loss/valid_length:.3f} | '
                  f'Valid accuracy: {valid_accuracy/valid_length:.3f}')

    print('----- Training completed -----')
    print('Training Time:', (time.time() - start_time), 'secs')

    checkpoint = {
        'epochs': epochs,
        'learning_rate': learning_rate,
        'model_classifier': classifier,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': image_datasets(source_dir, 'train').class_to_idx
    }

    torch.save(checkpoint, checkpoint_path)
    print('----- Model saved to', checkpoint_path, '-----')


def data_dir(source_dir, stage):
    return source_dir + '/' + stage

def data_transforms(stage):
    pic_size = 224 # picture resolution is 224x224 pixels
    means = [0.485, 0.456, 0.406]
    standard_deviations = [0.229, 0.224, 0.225]

    if stage == 'train':
        return transforms.Compose([transforms.RandomResizedCrop(pic_size, scale=(0.08, 1.0)), 
                            transforms.RandomHorizontalFlip(), 
                            transforms.ToTensor(), 
                            transforms.Normalize(means, standard_deviations)])
    elif stage == 'valid':
        return transforms.Compose([transforms.Resize(256), 
                            transforms.CenterCrop(pic_size), 
                            transforms.ToTensor(), 
                            transforms.Normalize(means, standard_deviations)]) 
    elif stage == 'test':
        return transforms.Compose([transforms.Resize(256), 
                            transforms.CenterCrop(pic_size), 
                            transforms.ToTensor(), 
                            transforms.Normalize(means, standard_deviations)])

def image_datasets(source_dir, stage):
    image_dataset = datasets.ImageFolder(data_dir(source_dir, stage), transform = data_transforms(stage))
    return image_dataset

def dataloaders(source_dir, stage):
    dataloader = utils.data.DataLoader(image_datasets(source_dir, stage), batch_size=64, shuffle=True)
    return dataloader

def main():
    input_args = get_input_args()
    models_arch = input_args.arch
    epochs = input_args.epochs
    learning_rate = input_args.learning_rate
    hidden_units = input_args.hidden_units # 512
    gpu = input_args.gpu
    source_dir = input_args.source_dir
    checkpoint_path = input_args.checkpoint_path
    hyperparameters  = [models_arch, epochs, learning_rate, hidden_units, gpu, source_dir, checkpoint_path]

    print('----- The Input Arguments -----')
    print('pre-trained model --arch:', models_arch)
    print('epochs --epochs:', epochs)
    print('learning_rate --learning_rate:', learning_rate)
    print('hidden_units --hidden_units:', hidden_units)
    print('using GPU/CPU --gpu:', gpu)
    print('source_dir --source_dir:', source_dir)
    print('checkpoint_path --save:', checkpoint_path)
    print('Start training using these settings above...')

    train(hyperparameters)


if __name__ == '__main__':
    main()