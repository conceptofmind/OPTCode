import torch

import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.core import global_context as gpc
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.nn.optimizer import HybridAdam

from utils.build_dataloader import build_dataloaders
from utils.huggingface_cfg import CFG

from transformers import OPTForCausalLM

def OPTCode(cfg: CFG):

    colossalai.launch_from_torch(config='./utils/colossalai_config.py')

    logger = get_dist_logger()

    assert hasattr(gpc.config, "EPOCHS"), "Please provide EPOCHS in your configuration"
    assert hasattr(gpc.config, "LEARNING_RATE"), "Please provide LEARNING_RATE in your configuration"
    assert hasattr(gpc.config, "gradient_accumulation"), "Please provide gradient_accumulation in your configuration"
    assert hasattr(gpc.config, "clip_grad_norm"), "Please provide clip_grad_norm in your configuration"

    if hasattr(gpc.config, "zero"):
        with ZeroInitContext(
            target_device = torch.cuda.current_device(),
            shard_strategy = gpc.config.zero.model_config.shard_strategy,
            shard_param = True
        ):
            model = OPTForCausalLM.from_pretrained("facebook/opt-1.3b")
            model.gradient_checkpointing_enable()
    else:
        model = OPTForCausalLM.from_pretrained("facebook/bloom-1.3b")

    # build dataloaders
    train_dataloader, eval_dataloader = build_dataloaders(cfg)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=gpc.config.LEARNING_RATE
    )

    #initialize the model
    
    engine, train_dataloader, eval_dataloader, _ = colossalai.initialize(
        model,
        optimizer,
        None,
        train_dataloader,
        eval_dataloader
    )

    steps = 0

    # training loop
    for _ in range(gpc.config.EPOCHS):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}
            
            engine.zero_grad()
            output = model(**batch)
            
            loss = output.loss
            
            engine.backward(loss)
            engine.step()

            steps += 1

            # validation loop    
            # engine.eval()
            # for step, batch in enumerate(eval_dataloader):
            # batch = {k: v.cuda() for k, v in batch.items()}
            #     with torch.no_grad():

            #         output = model(**batch)
            #     eval_loss = output.loss

OPTCode(CFG())