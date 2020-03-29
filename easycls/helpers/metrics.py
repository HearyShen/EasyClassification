import torch


def accuracy(outputs, targets, topk=[1]):
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
        self.total = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        """
        Update the class's confusion matrix

        Args:
            outputs: Tensor, model's outputs, [batch_size, model_output_size]
            targets: Tensor, ground-truth targets, [batch_size]
        
        Returns:
            the updated instance
        """
        with torch.no_grad():
            values, indices = outputs.max(
                dim=1)  # values and indices, [batch_size]
            # True: index = class_index, False: index != class_index
            batch_size = targets.size(0)
            C_expand = torch.tensor(self.class_index).expand_as(targets).to(
                outputs.device)
            AP_bools = targets.eq(C_expand)  # Boolean[batch_size]
            AN_bools = AP_bools.logical_not()
            PP_bools = indices.eq(C_expand)  # Boolean[batch_size]
            PN_bools = PP_bools.logical_not()

            # compute ConfusionMatrix
            # torch.mul computes logical AND for BoolTensor
            self.TP += PP_bools.mul(AP_bools).int().sum().item()
            self.FP += PP_bools.mul(AN_bools).int().sum().item()
            self.FN += PN_bools.mul(AP_bools).int().sum().item()
            self.TN += PN_bools.mul(AN_bools).int().sum().item()
            self.total = self.TP + self.FP + self.FN + self.TN

            # compute accuracy, precision, recall and F1 metrics
            self.accuracy = self.TP / self.total
            self.precision = self.TP / (self.TP + self.FP + 0.0001)
            self.recall = self.TP / (self.TP + self.FN + 0.0001)
            self.f1 = 2 * self.precision * self.recall / (self.precision +
                                                          self.recall + 0.0001)

        return self

    @staticmethod
    def update_all(confusion_matrices, outputs, targets):
        """
        Update all the confusion matrices with model's outputs and targets
        """
        for cmatrix in confusion_matrices:
            cmatrix.update(outputs, targets)

        return confusion_matrices

    @staticmethod
    def str_all(confusion_matrices):
        """
        Convert all the confusion matrices to string
        """
        cms_str = '\n'.join([str(cm) for cm in confusion_matrices])
        
        return cms_str

    @staticmethod
    def print_all(confusion_matrices):
        """
        Print all the confusion matrices
        """
        for cmatrix in confusion_matrices:
            print(cmatrix)

    def __str__(self):
        """
        Output the Confusion Matrix in format string

        e.g.

        ```
        Confusion Matrix of class 0
                AP      AN      Sum
        PP      16      15      31
        PN      31      66      97
        Sum     47      81
        Total: 128
        ```
        """
        cm_title = f'Confusion Matrix of class {self.class_index}'
        cm_line1 = f"\tAP\tAN\tSum"
        cm_line2 = f"PP\t{self.TP}\t{self.FP}\t{self.TP+self.FP}"
        cm_line3 = f"PN\t{self.FN}\t{self.TN}\t{self.FN+self.TN}"
        cm_line4 = f"Sum\t{self.TP+self.FN}\t{self.FP+self.TN}"
        cm_caption = f"Total: {self.total}\tAcc={self.accuracy*100:.3f}%\tPrec={self.precision*100:.3f}%\tRec={self.recall*100:.3f}%\tF1={self.f1:.3f}"

        cm_str = '\n'.join(
            [cm_title, cm_line1, cm_line2, cm_line3, cm_line4, cm_caption])
        return cm_str

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


# Unit Test
if __name__ == "__main__":
    batch_size = 16  # if batch_size, it is highly possible to encounter division-by-zero error (precision, recall, F1)
    class_count = 3

    outputs = torch.rand(batch_size, class_count).cuda()
    targets = torch.randint(0, class_count, [batch_size]).cuda()

    print(f'Outputs: {outputs}')
    print(f'Targets: {targets}')

    cm0 = ConfusionMatrix(0)
    cm0.update(outputs, targets)
    print(cm0)

    print(
        f'Acc={cm0.accuracy}, Prec={cm0.precision}, Recall={cm0.recall}, F1={cm0.f1}'
    )
