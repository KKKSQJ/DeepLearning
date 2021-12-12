# batch scalar record
def record_log(log_writer, losses, lr, batch_iter, batch_time, flag="Train", acc=None):
    log_writer.add_scalar(f"{flag}/batch_loss", losses.data.item(), batch_iter)
    log_writer.add_scalar(f"{flag}/learning_rate", lr, batch_iter)
    log_writer.add_scalar(f"{flag}/batch_time", batch_time, batch_iter)
    if acc is not None:
        log_writer.add_scalar(f"{flag}/batch_acc", acc, batch_iter)


def record_scalars(log_writer, mean_loss, epoch, flag="train", mean_acc=None):
    log_writer.add_scalar(f"{flag}/epoch_average_loss", mean_loss, epoch)
    if mean_acc is not None:
        log_writer.add_scalar(f"{flag}/epoch_average_acc", mean_acc, epoch)


def translate_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'module' in key:
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


class Metric_rank:
    def __init__(self, name):
        self.name = name
        self.sum = 0.0
        self.n = 0

    def update(self, val):
        self.sum += val
        self.n += 1

    @property
    def average(self):
        return self.sum / self.n
