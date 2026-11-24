import os
import logging
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None


class Logger:
    def __init__(self, log_dir, exp_name, use_tb=True):
        os.makedirs(log_dir, exist_ok=True)

        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"{exp_name}_{time_str}"

        # ========== logging ==========
        self.log_file = os.path.join(log_dir, self.exp_name + ".log")
        self.logger = logging.getLogger(self.exp_name)
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        fh = logging.FileHandler(self.log_file)
        fh.setFormatter(formatter)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

        # ========== TensorBoard ==========
        self.use_tb = use_tb and SummaryWriter is not None
        self.tb_writer = None
        if self.use_tb:
            tb_dir = os.path.join(log_dir, "tensorboard", self.exp_name)
            self.tb_writer = SummaryWriter(tb_dir)

        self.logger.info(f"Experiment: {self.exp_name}")
        self.logger.info(f"Log file: {self.log_file}")
        if use_tb and SummaryWriter is None:
            self.logger.warning("TensorBoard is unavailable in the current environment, so TB logging is disabled.")

    def log(self, msg):
        self.logger.info(msg)

    def tb_add_scalar(self, tag, value, step):
        if self.use_tb:
            self.tb_writer.add_scalar(tag, value, step)

    def tb_add_scalars(self, tag, value_dict, step):
        if self.use_tb:
            self.tb_writer.add_scalars(tag, value_dict, step)

    def close(self):
        if self.use_tb and self.tb_writer is not None:
            self.tb_writer.close()
