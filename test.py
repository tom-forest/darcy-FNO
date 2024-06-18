from neuralop.models import TFNO
from neuralop.losses.data_losses import LpLoss, H1Loss
from neuralop.training.trainer import Trainer
from load_dataset import load_samples
import torch
import matplotlib.pyplot as plt

device='cpu'

train_loader, test_loader, data_processor = load_samples(
        file_name='training_samples_128_fft.pt',
        n_train=1000, batch_size=32, 
        n_test=20,
        test_batch_size=32,
        positional_encoding=True,
        encode_input=True,
        encode_output=True
)
data_processor = data_processor.to(device)

model = TFNO(n_modes=(32, 32), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42, in_channels=7)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), 
                                lr=8e-3, 
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}


trainer = Trainer(model=model, n_epochs=20,
                  device=device,
                  data_processor=data_processor,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)

test_loaders = {64: test_loader}

trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)

test_samples = test_loader.dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = model(x.unsqueeze(0))

    ax = fig.add_subplot(3, 3, index*3 + 1)
    ax.imshow(x[0], cmap='gray')
    if index == 0: 
        ax.set_title('Input scalar permeability field')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 2)
    ax.imshow(y.squeeze())
    if index == 0: 
        ax.set_title('Ground-truth pressure field')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 3)
    ax.imshow(out.squeeze().detach().numpy())
    if index == 0: 
        ax.set_title('Model prediction pressure field')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
fig.show()
fig.savefig('wat2')

torch.save(model, "fft_model_v2.pt")