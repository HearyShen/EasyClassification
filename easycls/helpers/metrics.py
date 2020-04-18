import torch


ZERO = 1.0e-16

def accuracy(outputs: torch.Tensor, targets: torch.Tensor, topk=[1]):
    """
    Computes the accuracy over the k top predictions for the specified values of k

    Accuracy = (TP + TN) / (TP + FP + FN + TN)

    Args: 
        outputs: Tensor, size of [batch_size, model_output_size], the model's output.
        targets: Tensor, size of [batch_size], ground-truth target.
        topk:   list, requests the top-k accuracy.

    Returns:
        List, the top-k accuracy, e.g. [top1, top5]
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        values, indices = outputs.topk(
            maxk, dim=1, largest=True, sorted=True
        )  # values and indices from top1 to topk, [batch_size, maxk]
        indices = indices.t()  # transposed topk indices, [maxk, batch_size]
        correct = indices.eq(targets.view(
            1, -1).expand_as(indices))  # Boolean, [maxk, batch_size]

        res = []
        for k in topk:
            acc_k = correct[:k].view(-1).float().sum() / batch_size
            res.append(acc_k.item())  # accuracy in [0,1],float
        return res


class ConfusionMatrix:
    """
    ConfusionMatrix for binary classification class.

    In Multi-class classification problem, every class needs a confusion matrix to caculate precision, recall and F1.

    Confusion Matrix as:

    ```
        ┃  AP  AN
    ━━━━╋━━━━━━━━━━━
    PP  ┃  TP  FP
    PN  ┃  FN  TN
    ```

    AP(Actual Positive), AN(Actual Negative), PP(Predicted Positive), PN(Predicted Negative)

    TP(True Positive), FP(False Positive), FN(False Negative), TN(True Negative)

    Attributes:
        class_index: the index of the class
        TP: int, True Positive
        FP: int, False Positive
        FN: int, False Negative
        TN: int, True Negative
    """
    def __init__(self, class_index):
        self.class_index = class_index
        self.reset()

    def reset(self, default=0):
        # Actual/Predicted Positive/Negative
        self.AP = default
        self.AN = default
        self.PP = default
        self.PN = default
        # True/False Positive/Negative
        self.TP = default
        self.FP = default
        self.FN = default
        self.TN = default
        # etc.
        self.total = default
        self.accuracy = default
        self.precision = default
        self.recall = default
        self.f1 = default

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update the class's confusion matrix

        Args:
            predictions: Tensor, model's predictions, [batch_size]
            targets: Tensor, ground-truth targets, [batch_size]
        
        Returns:
            the updated instance
        """
        assert predictions.size() == targets.size()
        with torch.no_grad():
            # True: index = class_index, False: index != class_index
            C_expand = torch.tensor(self.class_index).expand_as(targets).to(
                predictions.device)
            AP_bools = targets.eq(C_expand)  # Boolean[batch_size]
            AN_bools = AP_bools.logical_not()
            PP_bools = predictions.eq(C_expand)  # Boolean[batch_size]
            PN_bools = PP_bools.logical_not()

            # basic statistics
            self.AP += AP_bools.sum().item()
            self.AN += AN_bools.sum().item()
            self.PP += PP_bools.sum().item()
            self.PN += PN_bools.sum().item()

            # compute ConfusionMatrix
            # torch.mul computes logical AND for BoolTensor
            self.TP += PP_bools.mul(AP_bools).sum().item()
            self.FP += PP_bools.mul(AN_bools).sum().item()
            self.FN += PN_bools.mul(AP_bools).sum().item()
            self.TN += PN_bools.mul(AN_bools).sum().item()
            self.total = self.TP + self.FP + self.FN + self.TN

            # compute accuracy, precision, recall and F1 metrics
            self.accuracy = (self.TP + self.TN) / self.total
            self.precision = self.TP / (self.TP + self.FP + ZERO)
            self.recall = self.TP / (self.TP + self.FN + ZERO)
            self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall + ZERO)

        return self

    def to_str(self, digits=4):
        """
        Convert the ConfusionMatrix instance to a table-format report str.

        e.g.

        ```
        Confusion Matrix of class '0':
                         AP         AN    Samples
             PP           1          8          9
             PN          10        109        119
        Samples          11        117

                   Accuracy  Precision     Recall   F1-score    Samples
        Class 0      0.0078     0.1111     0.0909     0.1000        128
        ```

        Args: 
            digits (int): digits of float to output. (default: 4)

        Returns:
            report (str): a table-format report str.
        """
        headers = ["AP", "AN", "Samples"]
        width = max(max(len(head) for head in headers), digits+2, 10)
        head_fmt = '{:>{width}s} ' + ' {:>{width}}' * len(headers) + '\n'
        row_fmt = '{:>{width}s} ' + ' {:>{width}}' * len(headers)

        rows = [
            ('PP', self.TP, self.FP, self.PP),
            ('PN', self.FN, self.TN, self.PN),
            ('Samples', self.AP, self.AN, '')
        ]

        report = f"Confusion Matrix of class '{self.class_index}':\n"
        report += head_fmt.format('', *headers, width=width)
        report += '\n'.join([row_fmt.format(*row, width=width) for row in rows]) + '\n\n'

        metrics_headers = ["Accuracy", "Precision", "Recall", "F1-score", "Samples"]
        metrics_head_fmt = '{:>{width}s} ' + ' {:>{width}}' * len(metrics_headers) + '\n'
        metrics_row_fmt = '{:>{width}s} ' + ' {:>{width}.{digits}f}' * 4 + ' {:>{width}}'
        metric_row = (f'Class {self.class_index}', self.accuracy, self.precision, self.recall, self.f1, self.total)

        report += metrics_head_fmt.format('', *metrics_headers, width=width)
        report += metrics_row_fmt.format(*metric_row, width=width, digits=digits)
        
        return report

    def __str__(self):
        """
        Convert the ConfusionMatrix instance to a table-format report str.
        """
        return self.to_str()

    def get_accuracy(self):
        """
        Compute the class accuracy of the model's outputs

        Accuracy = (TP + TN) / (TP + FP + FN + TN)

        Returns:
            Float, the accuracy of the current class
        """
        # self.accuracy = (self.TP + self.TN) / (self.TP + self.FP + self.FN + self.TN)

        return self.accuracy

    def get_precision(self):
        """
        Compute the class precision of the model's outputs

        Precision = TP / (TP + FP)

        Returns:
            Float, the precision of the current class
        """
        # self.precision = self.TP / (self.TP + self.FP)

        return self.precision

    def get_recall(self):
        """
        Compute the class recall of the model's outputs

        Recall = TP / (TP + FN)

        Returns:
            Float, the recall of the current class
        """
        # self.recall = self.TP / (self.TP + self.FN)

        return self.recall

    def get_f1(self):
        """
        Compute the class F1 of the model's outputs

        F1 = 2PR / (P + R)

        Returns:
            Float, the F1 of the current class
        """
        # self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)

        return self.f1


class MultiConfusionMatrices:
    """
    Confusion Martices for multi classification classes.

    Reference:
        https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1
    """

    def __init__(self, num_classes):
        self.cms = [ConfusionMatrix(i) for i in range(num_classes)]
        self.num_classes = num_classes
        self.reset()

    def reset(self, default=0):
        self.TP = default
        self.total = default
        self.macro_precision = default
        self.macro_recall = default
        self.macro_f1 = default
        self.weighted_precision = default
        self.weighted_recall = default
        self.weighted_f1 = default
        self.micro_precision = default
        self.micro_recall = default
        self.micro_f1 = default
        self.accuracy = default

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update all the confusion matrices with model's outputs and targets
        """
        assert predictions.size() == targets.size()
        with torch.no_grad():
            self.reset()    # NOTE: ConfusionMatrix has already accumulated values
            for cm in self.cms:
                # update binary confusion matrix
                cm.update(predictions, targets)

                # update macro, weighted and macro metrics
                # sum up macro metrics
                self.macro_precision += cm.precision
                self.macro_recall += cm.recall
                self.macro_f1 += cm.f1

                # sum up weighted metrics
                self.weighted_precision += cm.precision * cm.AP
                self.weighted_recall += cm.recall * cm.AP
                self.weighted_f1 += cm.f1 * cm.AP
                self.total += cm.AP

                # sum up micro metrics (accuracy)
                self.TP += cm.TP

            # average macro metrics
            self.macro_precision /= self.num_classes
            self.macro_recall /= self.num_classes
            self.macro_f1 /= self.num_classes

            # average weighted metrics
            self.weighted_precision /= self.total
            self.weighted_recall /= self.total
            self.weighted_f1 /= self.total

            # micro precision == micro recall == micro f1 == accuracy
            self.micro_precision = self.micro_recall = self.micro_f1 = self.accuracy = self.TP / self.total

        return self

    def to_str(self, digits=4):
        """
        Convert the MultiConfusionMatrices instance to table-format report str.

        Args: 
            digits (int): digits of float to output. (default: 4)

        Returns:
            report (str): a table-format report str.
        """
        headers = ["Accuracy", "Precision", "Recall", "F1-score", "Samples"]
        width = max(max(len(head) for head in headers), digits+2, 10)
        head_fmt = '{:>{width}s} ' + ' {:>{width}}' * len(headers) + '\n'
        row_fmt = '{:>{width}s} ' + ' {:>{width}.{digits}f}' * 4 + ' {:>{width}}'
        rows_classes = [
            (str(cm.class_index), cm.accuracy, cm.precision, cm.recall, cm.f1, cm.AP) for cm in self.cms
        ]
        rows_conclusions = [
            ('Micro', self.accuracy, self.micro_precision, self.micro_recall, self.micro_f1, self.total),
            ('Weighted', float('nan'), self.weighted_precision, self.weighted_recall, self.weighted_f1, self.total),
            ('Macro', float('nan'), self.macro_precision, self.macro_recall, self.macro_f1, self.total)
        ]

        report = head_fmt.format('', *headers, width=width)
        report += '\n'.join([row_fmt.format(*row, width=width, digits=digits) for row in rows_classes])
        report += '\n\n'
        report += head_fmt.format('', *headers, width=width)
        report += '\n'.join([row_fmt.format(*row, width=width, digits=digits) for row in rows_conclusions])
        return report
    
    def __str__(self):
        """
        Convert all the confusion matrices to string
        """
        return self.to_str()

# Unit Test
if __name__ == "__main__":

    batch_size = 64  # if batch_size, it is highly possible to encounter division-by-zero error (precision, recall, F1)
    class_count = 10

    acc_sum = 0
    acc = 0
    total = 0
    mcms = MultiConfusionMatrices(class_count)
    for i in range(2):
        outputs = torch.rand(batch_size, class_count)
        # predictions = torch.randint(0, class_count, [batch_size])
        scores, predictions = outputs.max(dim=1)
        targets = torch.randint(0, class_count, [batch_size])
        print(f'Predictions: {predictions}\nTargets: {targets}')

        mcms.update(predictions, targets)
        print(mcms)

        accs = accuracy(outputs, targets)
        acc_sum += accs[0] * batch_size
        total += batch_size
        acc1 = acc_sum / total
        
        print(acc1, total)

        assert mcms.accuracy == acc1

    print(mcms.cms[0])
