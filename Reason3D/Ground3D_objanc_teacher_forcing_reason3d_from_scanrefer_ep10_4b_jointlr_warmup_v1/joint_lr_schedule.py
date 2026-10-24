"""Stage-aware learning rates; preserve Answer updates and optimizer state."""
import math
GROUP_KEYS={
    "structured_tokens":("structured_token_lr","joint_structured_token_lr"),
    "qwen_lora":("qwen_lora_lr","joint_qwen_lora_lr"),
    "ground_visual":("visual_lr","joint_visual_lr"),
    "adapters":("adapter_lr","joint_adapter_lr"),
    "decoder":("decoder_lr","joint_decoder_lr"),
}
NEWLY_UNFROZEN={"ground_visual","adapters","decoder"}

def learning_rate(name,step,steps_per_epoch,training):
    if name not in GROUP_KEYS:raise ValueError("Unknown optimizer group: "+name)
    if step<0 or steps_per_epoch<1:raise ValueError("Invalid optimizer step count")
    answer=int(training["answer_warmup_epochs"])*steps_per_epoch
    total=int(training["epochs"])*steps_per_epoch
    joint=total-answer;warmup=int(training["joint_warmup_steps"])
    minimum=float(training["cosine_min_lr_ratio"])
    if joint!=int(training["joint_epochs"])*steps_per_epoch or not 0<warmup<joint or not 0<minimum<=1:
        raise ValueError("Invalid two-stage learning-rate schedule")
    base_key,joint_key=GROUP_KEYS[name]
    base,target=float(training[base_key]),float(training[joint_key])
    if not 0<target<=base:raise ValueError("Joint target must be positive and no greater than Answer base")
    if step<answer:
        return base*(minimum+(1-minimum)*.5*(1+math.cos(math.pi*step/total)))
    j=min(step-answer,joint)
    if name in NEWLY_UNFROZEN:
        if j<warmup:return target*(j+1)/warmup
        progress=(j-warmup)/(joint-warmup)
    else:progress=j/joint
    return target*(minimum+(1-minimum)*.5*(1+math.cos(math.pi*progress)))

def scheduler_for(optimizer,steps_per_epoch,training):
    from torch.optim.lr_scheduler import LambdaLR
    names=[group["name"] for group in optimizer.param_groups]
    if len(names)!=len(set(names)) or set(names)!=set(GROUP_KEYS):
        raise ValueError("Expected exact native no-Interactor optimizer groups")
    lambdas=[]
    for group in optimizer.param_groups:
        name=group["name"];base_key,_=GROUP_KEYS[name];base=float(training[base_key])
        if float(group["lr"])!=base:raise ValueError("Optimizer base learning rate differs")
        lambdas.append(lambda step,name=name,base=base:learning_rate(name,step,steps_per_epoch,training)/base)
    return LambdaLR(optimizer,lambdas)
