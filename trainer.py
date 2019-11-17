import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class TrainingConfig:
    def __init__(self):
        self.input_list_trn = None
        self.truth_list_trn = None
        self.input_list_dev = None
        self.truth_list_dev = None
        self.prediction_func = None
        self.loss_calc_func = None
        self.loss_funcs = []
        self.eval_funcs = []
        self.batch_size = 128
        self.epoch = 300
        self.epoch_per_validation = 1
        self.session_name = "session"

    def set_input_trn(self, input_list):
        self.input_list_trn = input_list

    def set_truth_trn(self, truth_list):
        self.truth_list_trn = truth_list

    def set_input_dev(self, input_list):
        self.input_list_dev = input_list

    def set_truth_dev(self, truth_list):
        self.truth_list_dev = truth_list

    def set_prediction_func(self, prediction_func):
        self.prediction_func = prediction_func

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_session_name(self, session_name):
        self.session_name = session_name

    def set_loss_calc_func(self, loss_calc_func):
        self.loss_calc_func = loss_calc_func

    def add_loss_func(self, loss_func):
        self.loss_funcs.append(loss_func)

    def add_eval_func(self, eval_func):
        self.eval_funcs.append(eval_func)


class TrainingState:
    def __init__(self, task_num=1):
        self.task_num = task_num

        # Epoch data
        self.it_no = 0
        self.total_it = 0
        self.loss_list: list = []
        self.pred_batch: list = []
        self.gold_batch: list = []
        self.current_data_size: int = 0
        self.running_loss: list = [0 for _ in range(self.task_num)]
        self.historical_pred_batches: list = []

        # Global data
        self.current_epoch_id: int = 0
        self.best_accuracy: list = [0 for _ in range(self.task_num)]
        self.best_global_loss: float = 1e5

    def set_data_size(self, data_size):
        self.current_data_size = data_size

    def set_running_loss(self, running_loss):
        self.running_loss = running_loss

    def set_pred_batch(self, pred_batch):
        self.pred_batch = pred_batch

    def set_gold_batch(self, gold_batch):
        self.gold_batch = gold_batch

    def record_pred_batch(self, pred_batch):
        self.historical_pred_batches.append(pred_batch)

    def clear_epoch_session(self):
        self.loss_list: list = []
        self.pred_batch: list = []
        self.gold_batch: list = []
        self.current_data_size: int = 0
        self.running_loss: list = [0 for _ in range(self.task_num)]
        self.historical_pred_batches: list = []

    def update_epoch(self):
        for i in range(self.task_num):
            epoch_loss = self.running_loss[i] / self.current_data_size
            print('Task-{} Epoch-{} Loss: {:.4f}'.format(i, self.current_epoch_id, epoch_loss))

        self.clear_epoch_session()
        self.current_epoch_id += 1


class Metrics:
    def __init__(self):
        self.bi_cls_metric_tag = ['acc', 'FPR', 'F1', 'R', 'P', 'NP', 'PR', 'MaxP', 'MinP', 'MeanP']
        self.mul_cls_metric_tag = ['cls', 'acc', 'la', 'F1', 'R', 'P', 'NP', 'PR']

    def bi_cls_metric(self, output, label):
        threshold = 0.5
        output = torch.sigmoid(output)
        max_prob = output.max().item()
        min_prob = output.min().item()
        mean_prob = output.mean().item()

        assert output.shape == label.shape, ("Pred:", output.shape, "Gold:", label.shape)

        pred = (output >= threshold)
        truth = (label >= threshold)
        right = pred.eq(truth)
        acc = right.sum().double().item() / right.numel()  # Accuracy

        tp = (truth * right).sum().double().item()
        tn = (~truth * right).sum().double().item()
        fp = (truth * ~right).sum().double().item()
        fn = (~truth * ~right).sum().double().item()

        p_num = tp + fp
        n_num = tn + fn
        pr = p_num / truth.numel()  # Positive Rate

        fpr = fp / (tn + fp) if tn + fp > 0 else 0  # False Positive Rate
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        n_precision = tn / n_num if n_num > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0
        metric = [acc, fpr, f1, recall, precision, n_precision, pr, max_prob, min_prob, mean_prob]

        for tag, val in zip(self.bi_cls_metric_tag, metric):
            print("%s: %.2f" % (tag, val), end=" | ")
        print()

        return acc

    def mul_cls_metric(self, output, label):
        truth = None
        cls_num = output.shape[-1]
        pred = torch.softmax(output, dim=-1)
        pred = pred.argmax(dim=-1, keepdim=True).view(-1, )
        label = label.type(torch.long).view(-1, )

        assert pred.shape == label.shape, ("Error: unequal shape between pred and label: ", pred.shape, label.shape)

        right = pred.eq(label)
        acc = right.sum().double().item() / right.numel()

        tps, tns, fps, fns = [], [], [], []
        for i in range(cls_num):
            truth = (label == i)
            tps.append((truth * right).sum().double().item())
            tns.append((~truth * right).sum().double().item())
            fps.append((truth * ~right).sum().double().item())
            fns.append((~truth * ~right).sum().double().item())

        for idx, (tp, tn, fp, fn) in enumerate(zip(tps, tns, fps, fns)):

            p_num = tp + fp  # positive sample rate
            n_num = tn + fn  #
            pr = p_num / truth.numel()

            acc_c = (tp + tn) / (tp + tn + fp + fn)
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            n_precision = tn / n_num if n_num > 0 else 0
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

            metric = [idx, acc, acc_c, f1, recall, precision, n_precision, pr]
            for tag, val in zip(self.mul_cls_metric_tag, metric):
                print("%s: %.2f" % (tag, val), end=" | ")
            print()

        return acc

    @staticmethod
    def evaluate_mse_task(pred, gold):
        gold = gold.squeeze(1)
        pred = pred.squeeze(1)
        for i, [img_p, img_g] in enumerate(zip(pred, gold)):
            img = torch.cat([img_p, img_g], dim=-1)
            img_name = "./res/%d.png" % i
            plt.imsave(img_name, img.numpy(), vmin=0, vmax=1)
        return 0


class Loss:
    def __init__(self):
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def calc_bce_loss(self, pred, gold):
        gold = gold.view(-1, )
        pred = pred.view(-1, )
        loss = self.bce_loss(pred, gold)
        return loss

    def calc_mce_loss(self, pred, gold):
        numel = gold.numel()
        gold = gold.view(numel)
        pred = pred.view(numel, -1)
        loss = self.mce_loss(pred, gold)
        return loss

    def calc_mse_loss(self, pred, gold):
        loss = self.mse_loss(pred, gold)
        return loss



class Trainer:
    def __init__(self, model, gpu_num=0, **kwargs):
        super(Trainer, self).__init__(**kwargs)

        self.training_config = None
        self.training_state = None
        self.device = torch.device("cuda:%d" % gpu_num if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        self.model = model
        self.model.to(self.device)
        self.model_path = "./model/"

        self.metrics = Metrics()
        self.loss_pool = Loss()

    def train(self):
        epoch_per_val = self.training_config.epoch_per_validation
        for epoch_id in range(self.training_config.epoch):
            inputs = self.training_config.input_list_trn
            truths = self.training_config.truth_list_trn
            batch_size = self.training_config.batch_size
            self.model.train()
            
            for batch_inputs, batch_truth in self.batch_generator(inputs, truths, batch_size):
                pred = self.training_config.prediction_func(*batch_inputs)

                self.training_state.set_pred_batch(pred)
                self.training_state.set_gold_batch(batch_truth)
                self.calculate_loss()
                self.back_propagation()

            if (epoch_id+1) % epoch_per_val == 0:
                inputs = self.training_config.input_list_dev
                truths = self.training_config.truth_list_dev
                self.model.eval()
            
                for batch_inputs, batch_truth in self.batch_generator(inputs, truths, batch_size):
                    pred = self.training_config.prediction_func(*batch_inputs)
                    pred = [e.cpu().detach() for e in pred]
                    self.training_state.record_pred_batch(pred)
                self.evaluate()

            self.training_state.update_epoch()

    def batch_generator(self, inputs, truths, batch_size):
        total_num = inputs[0].shape[0]
        is_rest = (total_num % batch_size) != 0
        batch_num = total_num // batch_size + int(is_rest)
        self.training_state.total_it = batch_num
        for batch_id in range(batch_num):
            self.training_state.it_no = batch_id + 1

            st = batch_id * batch_size
            end = st + batch_size
            batch_inputs = [comp[st:end].to(self.device) for comp in inputs]
            batch_truths = [comp[st:end].to(self.device) for comp in truths]

            yield batch_inputs, batch_truths

    def evaluate(self):
        sample_element = self.training_state.historical_pred_batches[0]
        element_type = type(sample_element)
        element_num = len(sample_element)
        assert element_type == list, element_type

        # Concatenate all prediction batches
        pred_batches = [[] for _ in range(element_num)]
        for tasks_pred in self.training_state.historical_pred_batches:
            for i, pred in enumerate(tasks_pred):
                pred_batches[i].append(pred)
        pred_batches = [torch.cat(pred, dim=0) for pred in pred_batches]

        # Evaluation
        tasks_acc = []
        golds = self.training_config.truth_list_dev
        eval_funcs = self.training_config.eval_funcs
        for pred, gold, eval_func in zip(pred_batches, golds, eval_funcs):
            acc = eval_func(pred, gold)
            tasks_acc.append(acc)
        print("Epoch-%d validated" % self.training_state.current_epoch_id)

        for i in range(self.training_state.task_num):
            model_name = self.model_path + "%s.task%d.best.param" % (self.training_config.session_name, i)
            if self.training_state.best_accuracy[i] < tasks_acc[i]:
                self.training_state.best_accuracy[i] = tasks_acc[i]
                print("The best acc is: %.5f at epoch %d" % (self.training_state.best_accuracy[i],
                                                             self.training_state.current_epoch_id))
                # torch.save(self.model, model_name)
                print("Model %s saved" % model_name)

    def calculate_loss(self):
        loss_list = []
        pred_list = self.training_state.pred_batch
        gold_list = self.training_state.gold_batch
        loss_func_list = self.training_config.loss_funcs
        for pred, gold, loss_func in zip(pred_list, gold_list, loss_func_list):
            loss = loss_func(pred, gold)
            loss_list.append(loss)

        self.training_state.loss_list = loss_list

        if (self.training_state.it_no % 50 == 0) or (self.training_state.it_no == self.training_state.total_it):
            print("'%s' Epoch-%d it %d/%d" % (self.training_config.session_name,
                                              self.training_state.current_epoch_id,
                                              self.training_state.it_no,
                                              self.training_state.total_it), end=" ")
            for i in range(self.training_state.task_num):
                print("task_%d Loss: %5.2f " % (i, self.training_state.loss_list[i].item()), end=" ")
            print()

    def back_propagation(self):
        self.optimizer.zero_grad()
        loss = sum(self.training_state.loss_list)
        loss.backward()
        self.optimizer.step()
        self.training_state.current_data_size += self.training_config.batch_size
        for i in range(self.training_state.task_num):
            self.training_state.running_loss[i] += \
                self.training_state.loss_list[i].item() * self.training_config.batch_size

    def format_and_train(self, data_trn, data_dev):
        input_list_trn = [data_trn[0]]
        truth_list_trn = [data_trn[1], data_trn[0]]
        input_list_dev = [data_dev[0]]
        truth_list_dev = [data_dev[1], data_dev[0]]

        training_config = TrainingConfig()
        self.training_config = training_config

        self.training_config.set_batch_size(128)
        self.training_config.set_epoch(20)
        self.training_config.set_input_trn(input_list_trn)
        self.training_config.set_truth_trn(truth_list_trn)
        self.training_config.set_input_dev(input_list_dev)
        self.training_config.set_truth_dev(truth_list_dev)
        self.training_config.set_prediction_func(self.model.forward)
        self.training_config.set_session_name("mnist")

        self.training_config.add_loss_func(self.loss_pool.calc_mce_loss)
        self.training_config.add_eval_func(self.metrics.mul_cls_metric)

        self.training_config.add_loss_func(self.loss_pool.calc_mse_loss)
        self.training_config.add_eval_func(self.metrics.evaluate_mse_task)

        self.training_state = TrainingState(task_num=2)

        self.train()
