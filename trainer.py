import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random


class TrainingConfig:
    def __init__(self):
        self.batch_size = 128
        self.epoch = 300
        self.session_name = "session"
        self.epoch_per_validation = 1
        self.input_list_trn = None
        self.truth_list_trn = None
        self.input_list_dev = None
        self.truth_list_dev = None
        self.input_list_test = None
        self.truth_list_test = None
        self.prediction_func = None
        self.__loss_funcs = []
        self.__eval_funcs = []

    def set_input_trn(self, input_list):
        self.input_list_trn = input_list

    def set_truth_trn(self, truth_list):
        self.truth_list_trn = truth_list

    def set_input_dev(self, input_list):
        self.input_list_dev = input_list

    def set_truth_dev(self, truth_list):
        self.truth_list_dev = truth_list

    def set_input_test(self, input_list):
        self.input_list_test = input_list

    def set_truth_test(self, truth_list):
        self.truth_list_test = truth_list

    def set_prediction_func(self, prediction_func):
        self.prediction_func = prediction_func

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_session_name(self, session_name):
        self.session_name = session_name

    def add_loss_func(self, loss_func):
        self.__loss_funcs.append(loss_func)

    def get_loss_funcs(self):
        return self.__loss_funcs

    def add_eval_func(self, eval_func):
        self.__eval_funcs.append(eval_func)

    def get_eval_funcs(self):
        return self.__eval_funcs


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
        self.best_model_path: str = ""
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

    def __clear_epoch_session(self):
        self.it_no = 0
        self.total_it = 0
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

        self.__clear_epoch_session()
        self.current_epoch_id += 1


class Metrics:
    def __init__(self):
        self.bi_cls_metric_tag = ['acc', 'F1', 'R', 'P', 'NP', 'PR', 'MaxP', 'MinP', 'MeanP']
        self.mul_cls_metric_tag = ['acc', 'F1', 'R', 'P', 'NP', 'PR']

    def bi_cls_metric(self, pred, truth):
        assert pred.shape == truth.shape, ("Pred:", pred.shape, "Gold:", truth.shape)

        pred = torch.sigmoid(pred)
        max_prob = pred.max().item()
        min_prob = pred.min().item()
        mean_prob = pred.mean().item()

        threshold = 0.5
        pred = (pred >= threshold)
        truth = (truth >= threshold)
        right = pred.eq(truth)
        acc, f1, recall, precision, n_precision, pr = self.calc_metric(truth, right)
        metric = [acc, f1, recall, precision, n_precision, pr, max_prob, min_prob, mean_prob]

        for tag, val in zip(self.bi_cls_metric_tag, metric):
            print("%s: %.2f" % (tag, val), end=" | ")
        print()

        return acc

    def mul_cls_metric(self, pred, truth):
        cls_num = pred.shape[-1]

        pred = torch.softmax(pred, dim=-1)
        pred = pred.argmax(dim=-1).view(-1, )
        truth = truth.type(torch.long)
        truth = truth.view(-1, )
        assert pred.shape == truth.shape, ("Error: unequal shape between pred and label: ", pred.shape, truth.shape)

        right = pred.eq(truth)
        acc = right.sum().double().item() / right.numel()

        print("Acc @ (all class): %.2f" % acc)
        for i in range(cls_num):
            mask = (truth == i).nonzero()
            local_pred = (pred[mask].view(-1, ) == i)
            local_truth = (truth[mask].view(-1, ) == i)
            local_right = local_pred.eq(local_truth)
            metric = self.calc_metric(local_truth, local_right)

            print("@(class %d)" % i, end=" | ")
            for tag, val in zip(self.mul_cls_metric_tag, metric):
                print("%s: %.2f" % (tag, val), end=" | ")
            print()

        return acc

    @staticmethod
    def calc_metric(truth, right):
        assert right.shape == truth.shape, ("Right:", right.shape, "Gold:", truth.shape)

        acc = right.sum().double().item() / right.numel()

        tp = (truth * right).sum().double().item()
        tn = (~truth * right).sum().double().item()
        fp = (truth * ~right).sum().double().item()
        fn = (~truth * ~right).sum().double().item()

        p_num = tp + fp
        n_num = tn + fn
        pr = p_num / truth.numel()  # Positive Rate

        recall = tp / (tp + fn) if tp + fn > 0 else 0
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        n_precision = tn / n_num if n_num > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0
        metric = [acc, f1, recall, precision, n_precision, pr]

        return metric

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
        batch_size = self.training_config.batch_size

        for epoch_id in range(self.training_config.epoch):
            # Training part
            inputs = self.training_config.input_list_trn
            truths = self.training_config.truth_list_trn
            self.model.train()
            
            for batch_inputs, batch_truth in self.batch_generator(inputs, truths, batch_size):
                batch_pred = self.training_config.prediction_func(*batch_inputs)

                self.training_state.set_pred_batch(batch_pred)
                self.training_state.set_gold_batch(batch_truth)
                self.calculate_loss()
                self.back_propagation()

            # Validation part
            if (epoch_id+1) % self.training_config.epoch_per_validation == 0:
                inputs = self.training_config.input_list_dev
                truths = self.training_config.truth_list_dev
                self.model.eval()
            
                for batch_inputs, batch_truth in self.batch_generator(inputs, truths, batch_size):
                    batch_pred = self.training_config.prediction_func(*batch_inputs)
                    batch_pred = [e.cpu().detach() for e in batch_pred]
                    self.training_state.record_pred_batch(batch_pred)
                self.evaluate()

            self.training_state.update_epoch()

        # Test part
        inputs = self.training_config.input_list_test
        truths = self.training_config.truth_list_test
        model_path = self.training_state.best_model_path
        self.model = torch.load(model_path)
        self.model.eval()

        for batch_inputs, batch_truth in self.batch_generator(inputs, truths, batch_size):
            batch_pred = self.training_config.prediction_func(*batch_inputs)
            batch_pred = [e.cpu().detach() for e in batch_pred]
            self.training_state.record_pred_batch(batch_pred)
        self.test()

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

    def calculate_loss(self):
        loss_list = []
        pred_list = self.training_state.pred_batch
        gold_list = self.training_state.gold_batch
        loss_func_list = self.training_config.get_loss_funcs()
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

    def evaluate(self):
        tasks_acc = self.__calc_metrics()

        # Show result and save model
        print("Epoch-%d validated;" % self.training_state.current_epoch_id)
        for i in range(self.training_state.task_num):
            acc = tasks_acc[i]
            print("Acc @ task %d: %.5f" % (i, acc))
            model_name = self.model_path + "%s.task%d.best.param" % (self.training_config.session_name, i)
            if self.training_state.best_accuracy[i] < tasks_acc[i]:
                self.training_state.best_accuracy[i] = tasks_acc[i]
                torch.save(self.model, model_name)
                self.training_state.best_model_path = model_name
                print("Best acc got! Model: %s saved" % model_name)

    def test(self):
        tasks_acc = self.__calc_metrics()

        # Show result and save model
        print("%s Tested;" % self.training_state.best_model_path)
        for i in range(self.training_state.task_num):
            acc = tasks_acc[i]
            print("Acc @ task %d: %.5f" % (i, acc))

    def __calc_metrics(self):
        # Concatenate all prediction batches
        pred_batches = [[] for _ in range(self.training_state.task_num)]
        for tasks_pred in self.training_state.historical_pred_batches:
            for i, pred in enumerate(tasks_pred):
                pred_batches[i].append(pred)
        pred_batches = [torch.cat(pred, dim=0) for pred in pred_batches]

        # Evaluation
        tasks_acc = []
        golds = self.training_config.truth_list_dev
        eval_funcs = self.training_config.get_eval_funcs()
        for pred, gold, eval_func in zip(pred_batches, golds, eval_funcs):
            acc = eval_func(pred, gold)
            tasks_acc.append(acc)

        return tasks_acc

    def prepare_and_train(self, data_trn, data_test):
        image_trn, label_trn = data_trn
        image_test, label_test = data_test

        trn_data_size = image_trn.shape[0]
        boundary = 50000
        indices = self.get_indices(trn_data_size)

        image_dev, label_dev = image_trn[indices[boundary:]], label_trn[indices[boundary:]]
        image_trn, label_trn = image_trn[indices[:boundary]], label_trn[indices[:boundary]]

        input_list_trn = [image_trn]
        truth_list_trn = [label_trn, image_trn]
        input_list_dev = [image_dev]
        truth_list_dev = [label_dev, image_dev]
        input_list_test = [image_test]
        truth_list_test = [label_test, image_test]

        self.training_config = TrainingConfig()
        self.training_state = TrainingState(task_num=2)

        # Global configure
        self.training_config.set_batch_size(128)
        self.training_config.set_epoch(50)
        self.training_config.set_input_trn(input_list_trn)
        self.training_config.set_truth_trn(truth_list_trn)
        self.training_config.set_input_dev(input_list_dev)
        self.training_config.set_truth_dev(truth_list_dev)
        self.training_config.set_input_test(input_list_test)
        self.training_config.set_truth_test(truth_list_test)
        self.training_config.set_prediction_func(self.model.forward)
        self.training_config.set_session_name("mnist")

        # Configure for the first task
        self.training_config.add_loss_func(self.loss_pool.calc_mce_loss)
        self.training_config.add_eval_func(self.metrics.mul_cls_metric)

        # Configure for the second task
        self.training_config.add_loss_func(self.loss_pool.calc_mse_loss)
        self.training_config.add_eval_func(self.metrics.evaluate_mse_task)

        self.train()

    @staticmethod
    def get_indices(size):
        random.seed(1)
        indices = list(range(size))
        random.shuffle(indices)
        indices = torch.Tensor(indices).type(torch.long)
        return indices
