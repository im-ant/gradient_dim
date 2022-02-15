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


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        # ==
        # Set up
        self.cfg = cfg
        # utils.set_seed_everywhere(cfg.seed)  # TODO implement this
        self.device = torch.device(cfg.device)

        # Logging
        self.logger = Logger(self.work_dir, use_tb=False, sort_header=False)

    def evaluate(self):
        pass

    def train(self):
        print(self.device)  # TODO delete

        vision_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_data = datasets.MNIST(
            root=self.cfg.dataset.parent_dir,
            train=True,
            transform=vision_transform,
            download=True,
        )
        test_data = datasets.MNIST(
            root=self.cfg.dataset.parent_dir,
            train=False,
            transform=vision_transform
        )

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=self.cfg.train.batch_size,
                                                   shuffle=True)

        # TODO delete this blob
        examples = enumerate(train_loader)
        batch_idx, (example_data, example_targets) = next(examples)
        print(example_data.shape)  #

        # Actual training below, TODO move above to initialization
        model = hydra.utils.instantiate(self.cfg.model)
        model.to(self.device)
        print(model)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)  # TODO improve this
        criterion = nn.CrossEntropyLoss()

        global_step = 0
        self.logger.log('train/step', 0, global_step)
        for epoch in range(1, self.cfg.train.epochs + 1):
            model.train()
            epoch_num_batch = 0
            epoch_examples = 0
            for batch_idx, (inputs, target) in enumerate(train_loader):
                inputs, target = inputs.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                #
                global_step += 1
                epoch_num_batch += 1
                epoch_examples += len(inputs)

                self.logger.log('train/loss', loss.item(), global_step)

                # Log
                if epoch_num_batch % self.cfg.logging.log_per_num_batch == 0:
                    # TODO: add timer
                    with self.logger.log_and_dump_ctx(global_step,
                                                      ty='train') as log:
                        log('step', global_step)
                        log('epoch', epoch)
                        log('batch_num', epoch_num_batch)
                        log('examples', epoch_examples)
                        # log('fps', episode_step / elapsed_time)
                        # log('total_time', total_time)
                        # log('step', self._global_step)

                # Save
                if epoch_num_batch % 2 == 0:  # TODO: make into config
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
