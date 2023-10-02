import wandb
import plotly.io as pio

def log_wandb_data(config_combination, wdn, algorithm, len_tra_database,
                   len_val_database, len_tst_database, cfg, train_config,
                   loss_plot, R2_plot):
    '''
    Function to prepare the data to be logged to wandb
    '''
    combination_to_log = config_combination.copy()
    combination_to_log.pop('indices')
    # wandb.config = combination I don't think this does anything?
    wandb.log({"Network": wdn, "Algorithm": algorithm})

    comb_keys = list(combination_to_log.keys())

    if "hid_channels" in comb_keys:
        wandb.log({"Hidden channels": combination_to_log["hid_channels"]})

    if "num_layers" in comb_keys:
        wandb.log({"Number of layers": combination_to_log["num_layers"]})

    if "num_blocks" in comb_keys:
        wandb.log({"Number of blocks": combination_to_log["num_blocks"]})

    wandb.log({"Number of outputs": combination_to_log["num_outputs"]})
    wandb.log({'Training samples': len_tra_database, "Validation samples": len_val_database,
               "Testing samples": len_tst_database})

    wandb.log({"Learning rate": cfg["adamParams"]["lr"],
              "Weight decay": cfg["adamParams"]["weight_decay"],
              "Number of epochs": cfg["trainParams"]["num_epochs"],
                "Batch size": cfg["trainParams"]["batch_size"],
               "Alpha loss": cfg["lossParams"]["alpha"]})

    wandb.log(train_config)
    wandb.log({"Loss": wandb.Image(loss_plot + ".png")})
    wandb.log({"R2": wandb.Image(R2_plot + ".png")})


# def log_wandb_graphs():
