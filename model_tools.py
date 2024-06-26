from neuralop.models import TFNO
from neuralop.losses.data_losses import LpLoss, H1Loss
from neuralop.training.trainer import Trainer
from load_dataset import load_samples
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

''' Multiple specialized tools to work with FNOs on darcy flow.'''

def model_train(model_file, dataset_file, n_train, n_test):
    """ Trains a model on n_train samples from dataset_file and saves it as model_file.
        It will use n_test samples for testing during the training.
        A 4:1 n_train:n_test ratio is usually good.
        File names should end in .pt (pytorch file extension)"""

    # change this to 'cuda' to switch to gpu processing on an nvidia card
    # may not work, and nvidia drivers must be installed
    device='cpu'

    train_loader, test_loader, data_processor = load_samples(
            file_name=dataset_file,
            n_train=n_train, batch_size=32, 
            n_test=n_test,
            test_batch_size=32,
            positional_encoding=True,
            encode_input=True,
            encode_output=True
    )
    data_processor = data_processor.to(device)

    # All training parameters copied from the orginal FNO paper
    model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42, in_channels=7)
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

    torch.save(model, model_file)


def model_test(model_file, dataset_file, n_train, n_test):
    ''' Evaluates the average relative error of the model on n_test samples.
        n_train is needed to avoid samples the model already knows.
        Using validation samples as testing samples is bad practice but requires less control over the dataloader.'''
    
    model = torch.load(model_file)

    # put model in evaluation mode. May or may not be properly implemented in the FNO library.
    model.eval()
    train_loader, test_loader, data_processor = load_samples(
            file_name=dataset_file,
            n_train=n_train, batch_size=30, 
            n_test=n_test,
            test_batch_size=32,
            positional_encoding=True,
            encode_input=True,
            encode_output=True
    )

    # the method exists for the data processor too but it doesnt work as expected - see next comment
    data_processor.eval()

    # dirty hackaround to validate a weird if statement in the FNO library
    # this may be proof I misused data loaders entirely, i'm very sorry
    data_processor.train = None

    avg = []
    for data in test_loader.dataset:
        x = np.copy(data["x"])
        y = np.copy(data["y"].squeeze())
        data = data_processor.preprocess(data, batched=False)
        xpre = data["x"]
        out = model(xpre.unsqueeze(0))
        out, data = data_processor.postprocess(out, data)
        out = out.squeeze().detach().numpy()

        err = np.linalg.norm(y - out)
        rel_err = err / np.linalg.norm(y)
        avg.append(rel_err)
    avg = np.array(avg)
    print("average relative error over test set: " + str(np.mean(avg)))


def model_plot(model_file, dataset_file, n_train, n_test, sample_index):
    """ Plots the evaluation of given model on given sample in the test set.
        Also contains repeated code, this is bad code i know."""

    model = torch.load(model_file)

    # put model in evaluation mode. May or may not be properly implemented in the FNO library.
    model.eval()
    train_loader, test_loader, data_processor = load_samples(
            file_name=dataset_file,
            n_train=n_train, batch_size=30, 
            n_test=n_test,
            test_batch_size=32,
            positional_encoding=True,
            encode_input=True,
            encode_output=True
    )

    # the method exists for the data processor too but it doesnt work as expected - see next comment
    data_processor.eval()

    # dirty hackaround to validate a weird if statement in the FNO library
    # this may be proof I misused data loaders entirely, i'm very sorry
    data_processor.train = None

    data = test_loader.dataset[sample_index]
    x = np.copy(data["x"])
    y = np.copy(data["y"].squeeze())
    data = data_processor.preprocess(data, batched=False)
    xpre = data["x"]
    out = model(xpre.unsqueeze(0))
    out, data = data_processor.postprocess(out, data)

    out = out.squeeze().detach().numpy()



    err = np.abs(y - out) / y

    fig, axs = plt.subplots(2, 2)
    errplot = axs[0,0].imshow(err, norm=matplotlib.colors.LogNorm())
    axs[0,0].set_title("relative error")
    fig.colorbar(errplot, ax=axs[0, 0])

    outplot = axs[0,1].imshow(out)
    axs[0,1].set_title("predicted pressure distribution")
    fig.colorbar(outplot, ax=axs[0, 1])

    yplot = axs[1,1].imshow(y)
    axs[1,1].set_title("true pressure distribution")
    fig.colorbar(yplot, ax=axs[1, 1])


    err_boundplot = axs[1,0].imshow(x[0])
    axs[1,0].set_title("input scalar permeability")
    fig.colorbar(err_boundplot, ax=axs[1, 0])


    fig.suptitle("FNO pressure prediction from permeability field and linear pressure boundary conditions")
    plt.show()