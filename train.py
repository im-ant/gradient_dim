# ==
#
#

from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets

from logger import Logger
import utils


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        # ==
        # Set up attributes
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # Logging
        self.timer = utils.Timer()
        self.logger = Logger(self.work_dir, use_tb=False, sort_header=False)

        # =====
        # Initialization
        vision_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.train_data = datasets.MNIST(
            root=self.cfg.dataset.parent_dir,
            train=True,
            transform=vision_transform,
            download=True,
        )
        self.test_data = datasets.MNIST(
            root=self.cfg.dataset.parent_dir,
            train=False,
            transform=vision_transform
        )

        # Initialize data loader and model
        self.train_loader = torch.utils.data.DataLoader(self.train_data,
                                                        batch_size=self.cfg.train.batch_size,
                                                        shuffle=True)

        self.model = hydra.utils.instantiate(self.cfg.model)
        self.model.to(self.device)
        print(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), **self.cfg.optimizer.kwargs)
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self):
        pass

    def train(self):
        global_step = 0
        self.logger.log('train/step', 0, global_step)
        for epoch in range(1, self.cfg.train.epochs + 1):
            model, optimizer, criterion = self.model, self.optimizer, self.criterion
            model.train()

            epoch_num_batch = 0
            epoch_examples = 0
            epoch_num_correct = 0
            fps_logging_steps = 0

            for batch_idx, (inputs, target) in enumerate(self.train_loader):
                inputs, target = inputs.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(inputs)  # size (mini-batch size, k classes)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                # Compute number correct for accuracy
                pred_label = output.max(axis=1).indices
                batch_n_correct = (pred_label == target).float().sum()
                epoch_num_correct += batch_n_correct

                #
                global_step += 1
                epoch_num_batch += 1
                epoch_examples += len(inputs)
                fps_logging_steps += 1

                self.logger.log('train/loss', loss.item(), global_step)

                # Log
                if epoch_num_batch % self.cfg.logging.log_per_num_batch == 0:
                    # TODO: compute eval accuracy?

                    elapsed_time, total_time = self.timer.reset()
                    cur_log_fps = fps_logging_steps / elapsed_time
                    fps_logging_steps = 0

                    with self.logger.log_and_dump_ctx(global_step,
                                                      ty='train') as log:
                        log('step', global_step)
                        log('epoch', epoch)
                        log('batch_num', epoch_num_batch)
                        log('examples', epoch_examples)
                        log('accuracy', epoch_num_correct / epoch_examples)
                        log('fps', cur_log_fps)
                        log('total_time', total_time)

                # Save
                if epoch_num_batch % int(self.cfg.logging.save_per_num_step) == 0:
                    cur_param_dict = {}
                    cur_grad_dict = {}
                    for name, param in model.named_parameters():
                        cur_param_dict[name] = param.detach().cpu()
                        cur_grad_dict[name] = param.grad.detach().cpu()

                    ppath = './ckpts'
                    Path(ppath).mkdir(parents=False, exist_ok=True)
                    torch.save(cur_param_dict, f'{ppath}/{global_step}_param_dict.pt')
                    torch.save(cur_grad_dict, f'{ppath}/{global_step}_grad_dict.pt')


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    print(cfg)

    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    workspace.train()


if __name__ == '__main__':
    main()
