from models.SlowFast_raw_signal_nsr_big import slowfast_raw
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from collections import OrderedDict
import torch
def load_train_objs(optim_name, lr, ft, pretrained_model, total_epochs, min_lr, num_channels,  add_FC,
                    input_size_seconds, dropout):
    # Load model
    model = slowfast_raw(dropout=dropout, num_channels=num_channels, layer_norm=False,
                             stochastic_depth=0, add_FC_BN=False, remove_BN=False, add_FC=add_FC,
                             replaceBN_LN=False, add_FC_LN=False, add_2FC=False,
                             input_size_seconds = input_size_seconds, slowfast_output=False, unet_output=True)

    if ft:
        state_dict = torch.load(pretrained_model)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[0:9] == "_orig_mod": # remove
                name = k[10:]
                new_state_dict[name] = v
            else:
                new_state_dict = state_dict
                break

        model.load_state_dict(new_state_dict, strict=True)

    if optim_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        lr_scheduler, _ = create_scheduler_v2(optimizer=optimizer, sched="cosine",
                                              num_epochs=total_epochs,
                                              warmup_epochs=0, min_lr=min_lr)
    elif optim_name == 'Adabelief':
        optimizer = create_optimizer_v2(model, opt="adabelief", lr=lr,
                                        weight_decay=1e-5, momentum=0.9)
        lr_scheduler, _ = create_scheduler_v2(optimizer=optimizer, sched="cosine",
                                              num_epochs=total_epochs,
                                              warmup_epochs=0, min_lr=min_lr)
    elif optim_name == 'lamb':
        optimizer = create_optimizer_v2(model, opt='lamb', lr=lr, weight_decay=1e-5)
        lr_scheduler, _ = create_scheduler_v2(optimizer=optimizer, sched="cosine",
                                              num_epochs=total_epochs,
                                              warmup_epochs=0, min_lr=min_lr)
    elif optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        lr_scheduler, _ = create_scheduler_v2(optimizer=optimizer, sched="cosine",
                                              num_epochs=total_epochs,
                                              warmup_epochs=0, min_lr=min_lr)
    elif optim_name == 'adamw':
        optimizer = create_optimizer_v2(model, opt='adamw', lr=lr, weight_decay=1e-5)
        lr_scheduler, _ = create_scheduler_v2(optimizer=optimizer, sched="cosine",
                                              num_epochs=total_epochs,
                                              warmup_epochs=0, min_lr=min_lr)
    elif optim_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.9, eps=1e-8)
        lr_scheduler, _ = create_scheduler_v2(optimizer=optimizer, sched="cosine",
                                              num_epochs=total_epochs,
                                              warmup_epochs=0, min_lr=min_lr)
    else:
        print("Indicate optimizer!!!!")

    return model, optimizer, lr_scheduler