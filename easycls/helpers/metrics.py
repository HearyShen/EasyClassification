import torch


def accuracy(outputs, targets, topk=(1, )):
    """
    Computes the accuracy over the k top predictions for the specified values of k

    Accuracy = (TP + TN) / (TP + FP + FN + TN)

    Args: 
        outputs: Tensor, size of [batch_size, model_output_size], the model's output.
        targets: Tensor, size of [batch_size], ground-truth target.
        topk:   tuple, requests the top-k accuracy.

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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 /
                                      batch_size))  # accuracy in [0,100] %
        return res


class ConfusionMatrix():
    """
    ConfusionMatrix for every class

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

    def reset(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0

    def update(self, outputs, targets):
        """
        Update the class's confusion matrix

        Args:
            outputs: Tensor, model's outputs, [batch_size, model_output_size]
            targets: Tensor, ground-truth targets, [batch_size]
        """
        with torch.no_grad():
            values, indices = outputs.max(dim=1)  # values and indices, [batch_size]
            # True: index = class_index, False: index != class_index
            batch_size = targets.size(0)
            C_expand = torch.tensor(self.class_index).expand_as(targets)
            AP_bools = targets.eq(C_expand)     # Boolean[batch_size]
            AN_bools = AP_bools.logical_not()
            PP_bools = indices.eq(C_expand)     # Boolean[batch_size]
            PN_bools = PP_bools.logical_not()

            # compute ConfusionMatrix
            self.TP = PP_bools.eq(AP_bools).int().sum().item()
            self.FP = PP_bools.eq(AN_bools).int().sum().item()
            self.FN = PN_bools.eq(AP_bools).int().sum().item()
            self.TN = PN_bools.eq(AN_bools).int().sum().item()

            # TODO: debug the error in this assertion
            assert batch_size == self.TP + self.FP + self.FN + self.TN

            # compute accuracy, precision, recall and F1 metrics
            self.accuracy = self.TP / batch_size
            self.precision = self.TP / (self.TP + self.FP)
            self.recall = self.TP / (self.TP + self.FN)
            self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)

    def get_accuracy(self):
        """
        Compute the class accuracy of the model's outputs

        Accuracy = TP / (TP + FP + FN + TN)

        Returns:
            Float, the accuracy of the current class
        """
        # self.TP + self.FP + self.FN + self.TN

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
        

if __name__ == "__main__":
    batch_size = 8
    class_count = 3

    outputs = torch.rand(batch_size, class_count)
    targets = torch.randint(0, class_count, [batch_size])

    cm0 = ConfusionMatrix(0)
    cm0.update(outputs, targets)

    print(f'Acc={cm0.accuracy}, Prec={cm0.precision}, Recall={cm0.recall}, F1={cm0.f1}')