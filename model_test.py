import torch
from load_dataset import load_samples
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

model = torch.load("fft_model_v2.pt")
model.eval()
train_loader, test_loader, data_processor = load_samples(
        file_name='training_samples_128_fft.pt',
        n_train=1000, batch_size=30, 
        n_test=31,
        test_batch_size=32,
        positional_encoding=True,
        encode_input=True,
        encode_output=True
)

data_processor.eval()
data_processor.train = None
data = test_loader.dataset[2]
x = np.copy(data["x"])
y = np.copy(data["y"].squeeze())
data = data_processor.preprocess(data, batched=False)
xpre = data["x"]
out = model(xpre.unsqueeze(0))
out, data = data_processor.postprocess(out, data)

out = out.squeeze().detach().numpy()



err = np.abs(y - out) / y
err_bound = np.abs(x[1] - out) / x[1]

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