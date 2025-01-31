import numpy as np
import torch
import csv
import pandas as pd
import os
import matplotlib.pyplot as plt

# base class for metrics
class Metric(object):
    def __init__(self, ini, updater, calculator):
        self.value = ini
        self.updater = updater
        self.calculator = calculator
    def update(self, target, pred):
        self.value += self.updater(target, pred)
    def calculate(self, count):
        self.value = self.calculator(self.value, count)
        return self.value


class LossMetric(object):
    def __init__(self, ini):
        self.value = ini
    def update(self, loss):
        self.value += loss
    def calculate(self, count):
        self.value = self.value / count
        return self.value    



# use once for every epoch
# have metrics
# write metrics for yourself
# 0. Do you need something except for the target and pred ?
#    Then rewrite Result and Metric.
# 1. define updater
# 2. define calculator
# 3. initialize the metric
class Result(object):
    def __init__(self):
        self.count = 0

        def accuracy_updater(target, pred):
            if target == np.argmax(pred):
                return 1
            else:
                return 0

        def accuracy_calculator(value, count):
            return value / count

        self.accuracy = Metric(0, accuracy_updater, accuracy_calculator)

        self.loss = LossMetric(0)

    def update(self, targets, preds, loss):
        self.loss.update(loss)

        for target, pred in zip(targets, preds):
            for key, value in self.__dict__.items():
                if value.__class__.__name__ == "Metric":
                    getattr(self, key).update(target, pred)
            self.count += 1

    def calculate(self):
        self.loss.calculate(self.count)

        for key, value in self.__dict__.items():
            if value.__class__.__name__ == "Metric" or value.__class__.__name__ == "LossMetric":
                getattr(self, key).calculate(self.count)

# use once for one training
# save and visualize the results of each epoch
class Logger(object):
    def __init__(self, place, result, file):
        self.file = file
        self.place = place
        self.fieldnames = []

        log_dict = {}

        for key, value in result.__dict__.items():
            if value.__class__.__name__ == "Metric" or value.__class__.__name__ == "LossMetric":
                log_dict[key] = value.value
                self.fieldnames.append(key)
                setattr(self, key, [value])

        with open(self.file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerow(log_dict)

    def append(self, result):
        log_dict = {}

        for key, value in result.__dict__.items():
            if value.__class__.__name__ == "Metric" or value.__class__.__name__ == "LossMetric":
                logs = getattr(self, key)
                logs.append(value)
                log_dict[key] = value.value
                setattr(self, key, logs)

        with open(self.file, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(log_dict)

    def write_into_file(self, name):
        df = pd.read_csv(self.file)
        values = df.values.T
        columns = df.columns
        epochs = np.array([i for i in range(len(values[0]))])

        for i, c in enumerate(columns):
            filename = os.path.join(self.place, "{}_{}.jpg".format(name, c))
            plt.figure()
            plt.plot(epochs, values[i])
            plt.savefig(filename)