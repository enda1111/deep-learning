import nn
from nn import models
import dataset_mnist

(x_train, t_train), (x_test, t_test) = dataset_mnist.load_mnist(flatten=False)

network = models.CNN()
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
