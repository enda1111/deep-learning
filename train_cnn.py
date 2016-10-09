import nn
from nn import models
import dataset_mnist
import dataset_cu

(x_train, t_train), (x_test, t_test) = dataset_cu.load_cu()#dataset_mnist.load_mnist(flatten=False)

network = models.CNN(input_dim=(3, 98, 66))
trainer = nn.Trainer(network, x_train, t_train, x_test, t_test,
                     epoch=20, mini_batch_size=100,
                     optimizer='Adam', optimizer_param={'lr': 0.001},
                     evaluate_sample_num_per_epoch=1000
                     )
print("Start Train")
trainer.train()
print("Finish Train")

network.save_params('cnn_params.pkl')
print("Saved Network Parameters")
