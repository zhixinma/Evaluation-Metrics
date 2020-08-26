from trainer import Trainer
from trainer import TrainingConfig

trainer = Trainer(the_model)
training_config = TrainingConfig()
training_config.set_data(input_trn, truth_trn, input_dev, truth_dev)
training_config.set_pad(input_pad, truth_pad)
training_config.set_forward_func(the_forward_func)
training_config.set_conf(batch_size=256, epoch=200, session_name=the_session_name)
training_config.add_task(loss_func=the_loss_func, eval_func=the_eval_func)
trainer.config(training_config)
