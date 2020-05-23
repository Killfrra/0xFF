import torch
from torch import nn
from torch.optim import SGD
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from PIL.ImageOps import autocontrast
from ignite.engine import Engine, Events
from ignite.metrics import MeanSquaredError, RunningAverage
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping
from torchsummary import summary

class Autoencoder(nn.Module):
    def __init__(self, ngpu):
        super(Autoencoder, self).__init__()
        self.ngpu = ngpu
        self.enable_decoder = True

        def down_conv_block(n_input, n_output, k_size=3, stride=1, padding=1):
            return [
                nn.Conv2d(n_input, n_output, k_size, stride, padding),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(n_output),
                nn.MaxPool2d(2)
            ]
        
        def up_conv_block(n_input, n_output, k_size=3, stride=1, padding=1):
            return [
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(n_input, n_output, k_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2, True), # negative_slope?
                nn.BatchNorm2d(n_output),
            ]
        
        self.encoder = nn.Sequential(
            * down_conv_block(1, 64),
            * down_conv_block(64, 64)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            * up_conv_block(64, 64),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        if self.enable_decoder:
            x = self.decoder(x)
        return x

image_size = 64
data_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.Lambda(autocontrast),
    transforms.ToTensor()
])

def get_data_loaders(train_batch_size, val_batch_size, workers):
    train_loader = DataLoader(datasets.ImageFolder('datasets/top5_real+synth', data_transform), batch_size=train_batch_size, shuffle=True, num_workers=workers)
    val_loader  =  DataLoader(datasets.ImageFolder('datasets/top5_synth_test', data_transform), batch_size=train_batch_size, shuffle=True, num_workers=workers)
    
    return train_loader, val_loader

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
ngpu = 1

model = Autoencoder(ngpu).to(device)

learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
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
    print('Epoch %d - Loss: %.2f' % (engine.state.epoch, engine.state.output))

train_batch_size = 128
val_batch_size = 128
num_workers = 6
train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size, num_workers)

@trainer.on(Events.EPOCH_COMPLETED)
def print_trainer_logs(engine):
    evaluator.run(val_loader, max_epochs=1)
    avg_loss = engine.state.metrics['loss']
    print('Epoch %d - Avg loss: %.2f' % (engine.state.epoch, avg_loss))

@trainer.on(Events.EPOCH_COMPLETED(every=5))
def check_accuracy(engine):
    evaluator.run(val_loader, max_epochs=1)

def accuracy_function(engine):
    return -engine.state.metrics['mse']

handler = EarlyStopping(1, accuracy_function, trainer)
evaluator.add_event_handler(Events.COMPLETED, handler)

def loss_function(engine):
    return -engine.state.metrics['loss']

to_save = { 'trainer': trainer, 'model': model, 'optimizer': optimizer }
handler = Checkpoint(to_save, DiskSaver('output/autoencoder', require_empty=False), score_function=loss_function, score_name='loss', n_saved=20)
trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

if handler.last_checkpoint:
    checkpoint = torch.load(handler.last_checkpoint)
    Checkpoint.load_objects(to_save, checkpoint)
"""
from PIL import Image
import torchvision.utils as vutils
inputs = data_transform(Image.open('datasets/top5_synth_test/MyriadPro-Regular/200.tiff')).unsqueeze_(0).to(device)
with torch.no_grad():
    outputs = model(inputs)
    vutils.save_image(outputs, 'ae_out.tiff', normalize=True)
exit()
"""
trainer.run(train_loader, max_epochs=100)


