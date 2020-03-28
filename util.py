import argparse
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def initialize_tensorboard(log_dir, common_name):
    """
    In distributed training, tensorboard doesn't work with multiple writers
    reference: https://stackoverflow.com/a/37411400/4569025
    """
    tb_log_path = Path(log_dir).joinpath(common_name)
    if not os.path.exists(tb_log_path):
        os.mkdir(tb_log_path)
    tb_writer = SummaryWriter(log_dir=tb_log_path)
    return tb_writer


def update_tensorboard(tb_writer, epoch, train_dict):
    """
    {"loss_rgb": mean_rgb, "loss_reg_rgb": mean_reg_rgb, "loss_depth": mean_depth,
        "loss_reg_depth": mean_reg_depth}
    """
    tb_writer.add_scalar(tag='RGB train loss', scalar_value=train_dict["loss_rgb"], global_step=epoch)
    tb_writer.add_scalar(tag='RGB regularized train loss', scalar_value=train_dict["loss_reg_rgb"], global_step=epoch)
    tb_writer.add_scalar(tag='Depth train loss', scalar_value=train_dict["loss_depth"], global_step=epoch)
    tb_writer.add_scalar(tag='Depth regularized train loss', scalar_value=train_dict["loss_reg_depth"], global_step=epoch)


def parse():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # config = configparser.ConfigParser()
    return args
