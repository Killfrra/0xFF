import torch
from torch import nn
from torch.optim import SGD
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Accuracy, Loss
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping
from encoder.model import Autoencoder
from model import DeepFont

learning_rate = 0.01
momentum = 0.9 #?

train_dataset = 'datasets/top60_ru_synth_train_preprocessed'
eval_dataset  = 'datasets/top60_ru_synth_test_preprocessed'
last_checkpoint = None #'output/classificator/checkpoint_loss=-0.015746485793698094.pth'
last_encoder_checkpoint = 'output/autoencoder/checkpoint_loss=-0.002959001278966386.pth'
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
    transforms.ToTensor()
])

def get_data_loaders(train_batch_size, val_batch_size, workers):
    train_loader = DataLoader(datasets.ImageFolder(train_dataset, data_transform), batch_size=train_batch_size, shuffle=True, num_workers=workers)
    val_loader  =  DataLoader(datasets.ImageFolder(eval_dataset, data_transform), batch_size=train_batch_size, shuffle=True, num_workers=workers)
    
    return train_loader, val_loader

encoder = Autoencoder(ngpu, enable_decoder=False) #.to(device)
model = DeepFont(encoder, ngpu)                   #.to(device)
optimizer = SGD(model.parameters(), learning_rate, momentum)
criterion = nn.CrossEntropyLoss()

trainer = create_supervised_trainer(model, optimizer, criterion, device)
metrics = { 'accuracy': Accuracy(), 'loss': Loss(criterion) }
evaluator = create_supervised_evaluator(model, metrics, device)

@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(engine):
    print('Epoch %d - Loss: %.4f' % (engine.state.epoch, engine.state.output))

def loss_function(engine):
    return -engine.state.metrics['loss']

to_save = { 'trainer': trainer, 'model': model, 'optimizer': optimizer }
handler = Checkpoint(to_save, DiskSaver('output/classifier', require_empty=False), score_function=loss_function, score_name='loss', n_saved=patience * check_interval)
trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

@trainer.on(Events.EPOCH_COMPLETED)
def print_trainer_logs(engine):
    avg_loss = engine.state.metrics['loss']
    print('Epoch %d - Avg loss: %.4f' % (engine.state.epoch, avg_loss))

train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size, num_workers)

@trainer.on(Events.EPOCH_COMPLETED(every=check_interval))
def check_accuracy(engine):
    evaluator.run(val_loader, max_epochs=1)

def accuracy_function(engine):
    accuracy = engine.state.metrics['accuracy']
    print('Epoch %d - Accuracy %.4f' % (engine.state.epoch, accuracy))
    return -accuracy

handler = EarlyStopping(round(patience / check_interval), accuracy_function, trainer)
evaluator.add_event_handler(Events.COMPLETED, handler)

encoder_checkpoint = torch.load(last_encoder_checkpoint)
Checkpoint.load_objects({ 'model': encoder }, encoder_checkpoint)
print(last_encoder_checkpoint, 'loaded')

if last_checkpoint:
    checkpoint = torch.load(last_checkpoint)
    Checkpoint.load_objects(to_save, checkpoint)
    print(last_checkpoint, 'loaded')

trainer.run(train_loader, max_epochs=10)