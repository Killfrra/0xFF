import torch
from torch import nn
from torch.optim import Adam
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
#from PIL.ImageOps import autocontrast
from ignite.engine import Engine, Events
from ignite.metrics import MeanSquaredError, RunningAverage
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping
from model import Autoencoder

#image_size = 96
learning_rate = 0.01

train_dataset = 'datasets/preprocessed_unlabeled_real'
eval_dataset = 'datasets/preprocessed_labeled_real'
last_checkpoint = 'output/autoencoder/checkpoint_loss=-0.015746485793698094.pth'
train_batch_size = 64
val_batch_size = 64
num_workers = 6

check_interval = 1
patience = 2
ngpu = 1

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda and ngpu > 0 else "cpu")

data_transform = transforms.Compose([
    transforms.Grayscale(),
    #transforms.Resize(image_size),
    #transforms.CenterCrop(image_size),
    #transforms.Lambda(autocontrast),
    transforms.ToTensor()
])

def get_data_loaders(train_batch_size, val_batch_size, workers):
    train_loader = DataLoader(datasets.ImageFolder(train_dataset, data_transform), batch_size=train_batch_size, shuffle=True, num_workers=workers)
    val_loader  =  DataLoader(datasets.ImageFolder(eval_dataset, data_transform), batch_size=train_batch_size, shuffle=True, num_workers=workers)
    
    return train_loader, val_loader

model = Autoencoder(ngpu).to(device)
optimizer = Adam(model.parameters(), learning_rate)
criterion = nn.MSELoss()

def process_function(engine, batch):
    inputs = batch[0].to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, inputs)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_function(engine, batch):
    model.eval()
    with torch.no_grad():
        inputs = batch[0].to(device)
        outputs = model(inputs)
        return outputs, inputs

trainer = Engine(process_function)
evaluator = Engine(evaluate_function)
"""
training_history = {'mse': []}
validation_history = {'mse': []}
"""
RunningAverage(output_transform=lambda x:x).attach(trainer, 'loss')
MeanSquaredError().attach(evaluator, 'mse')

@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(engine):
    print('Epoch %d - Loss: %.4f' % (engine.state.epoch, engine.state.output))

train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size, num_workers)

def loss_function(engine):
    return -engine.state.metrics['loss']

to_save = { 'trainer': trainer, 'model': model, 'optimizer': optimizer }
handler = Checkpoint(to_save, DiskSaver('output/autoencoder', require_empty=False), score_function=loss_function, score_name='loss', n_saved=patience * check_interval)
trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

@trainer.on(Events.EPOCH_COMPLETED)
def print_trainer_logs(engine):
    avg_loss = engine.state.metrics['loss']
    print('Epoch %d - Avg loss: %.4f' % (engine.state.epoch, avg_loss))

@trainer.on(Events.EPOCH_COMPLETED(every=check_interval))
def check_accuracy(engine):
    evaluator.run(val_loader, max_epochs=1)

def accuracy_function(engine):
    accuracy = engine.state.metrics['mse']
    print('Epoch %d - Loss %.4f' % (engine.state.epoch, accuracy))
    return -accuracy

handler = EarlyStopping(round(patience / check_interval), accuracy_function, trainer)
evaluator.add_event_handler(Events.COMPLETED, handler)

if last_checkpoint:
    checkpoint = torch.load(last_checkpoint)
    Checkpoint.load_objects(to_save, checkpoint)
    print(last_checkpoint, 'loaded')

trainer.run(train_loader, max_epochs=10)