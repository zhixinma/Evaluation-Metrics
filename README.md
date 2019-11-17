
## class: Loss  # A Series of Loss function  
 - func: calc_bce_loss
 - func: calc_mce_loss
 - func: calc_mse_loss

## class: Metrics  # A Series of Evaluation Metrics  
 - func: bi_cls_metric 
   - *Accuracy, Positive Rate, True Positive Rate, False Positive Rate, Recall, Positive Precision, Negative Precision, F1 score*  
 - func: mul_cls_metric 
   - *Accuracy, Positive Rate, True Positive Rate, False Positive Rate, Recall, Positive Precision, Negative Precision, F1 score*  
 - func: evaluate_mse_task  


## class: Trainer  # training process tempalte
 - func: train  
 - func: batch_generator  
 - func: calculate_loss  
 - func: back_propagation  
 - func: evaluate  
 - func: format_and_train  
 
