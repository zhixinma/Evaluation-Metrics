The class **Traininer** provides a template for neural networking training.

```python 
class: Trainer   
    def config()  # pucblic
    def train()  # pucblic
    def train_epoch()
    def validate()
    def test()
    def inference()
    def batch_generator()  
    def calculate_loss()  
    def back_propagation()
    def evaluate()  
    def calculate_metrics()  
    def batch_padding()
```

The class **Loss** and **Metrics** provide some common methods to calculate the loss and evaluation metrics. 

```python
class Loss
# A Series of Loss function  
    def calc_bce_loss()
    def calc_mce_loss()
    def calc_mse_loss()
```

```python
class Metrics  
# A Series of Evaluation Metrics  
    def bi_cls_metric()
        return Accuracy, Positive_Rate, True_Positive_Rate, False_Positive_Rate, Recall, Positive_Precision, Negative_Precision, F1_score 
    def mul_cls_metric()
        return Accuracy, Positive_Rate, True_Positive_Rate, False_Positive_Rate, Recall, Positive_Precision, Negative_Precision, F1_score
 - func: evaluate_mse_task  
```


Data Transfer Object: TrainingConfig and TrainingState

```python
class TrainingConfig
  def set_data()
  def set_pad()
  def set_conf()
  def set_forward_func()
  def add_task()
```

```python
class TrainingState
  def set_pred_batch()
  def set_gold_batch()
  def record_metric_batch()
  def clear_epoch_session()
  def get_best_model_path()
  def clear_infer_session()
  def update_epoch()
```
 
