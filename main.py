"""Main Code."""


from config import get_parameters
from train import train_network
from evaluate import evaluate_network


if __name__ == "__main__":
    args = get_parameters()

    if args.mode in ['train', 'resume']:
        if args.wandb:
            import wandb
            run = wandb.init(project=args.wb_project,
                             tags=args.wb_tags,
                             config=args)
        train_network(args)

        if args.wandb:
            wandb.alert(title=f"{run.name}",
                        text="training done!")
            wandb.finish()

    elif args.mode == 'eval':
        evaluate_network(args)
