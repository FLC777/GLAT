from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('anneal')
class AnnealingScheduler(FairseqLRScheduler):
    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)

    def step(self, epoch, val_loss=None):
        return self.optimizer.get_lr()

    @staticmethod
    def add_args(parser):
        parser.add_argument('--warmup-updates', default=0, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--anneal-steps', default=250000, type=int)
        parser.add_argument('--init-lr', type=float, default=3e-4)
        parser.add_argument('--end-lr', type=float, default=1e-5)

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if self.args.warmup_updates > 0 and num_updates <= self.args.warmup_updates:
            self.optimizer.set_lr(self.args.init_lr)
        else:
            args = self.args
            lr = max(0, (args.init_lr - args.end_lr) * (
                    args.anneal_steps - num_updates) / args.anneal_steps) + args.end_lr
            self.optimizer.set_lr(lr)
        return self.optimizer.get_lr()