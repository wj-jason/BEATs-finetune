import pytorch_lightning as pl
from datamodules.AercousticsDataModule import AercousticsDataModule
from fine_tune.transferLearning import BEATsTransferLearningModel

# Initialize the data module
data_module = AercousticsDataModule()
data_module.setup()

# Initialize the model
model = BEATsTransferLearningModel()

# Initialize a Trainer
trainer = pl.Trainer(accelerator='gpu', max_epochs = 10)

# Train the model
trainer.fit(model, datamodule=data_module)

# Test the model
test_results = trainer.test(model, datamodule=data_module)
print(test_results)
