# Pretrain loader adapted from Tencent's MedicalNet (Med3D) model.py:
# https://github.com/Tencent/MedicalNet
# Chen et al., "Med3D: Transfer Learning for 3D Medical Image Analysis", arXiv:1904.00625 (2019)

import torch
import resnet

def generate_model(model_name='resnet10', num_seg_classes=2, no_cuda=False, phase='train', pretrain_path=None, new_layer_names=['fc']):

    # Scaffold the right model based on user request
    if model_name == 'resnet10':
        model = resnet.resnet10(num_seg_classes=num_seg_classes)
    elif model_name == 'resnet18':
        model = resnet.resnet18(num_seg_classes=num_seg_classes)
    elif model_name == 'resnet34':
        model = resnet.resnet34(num_seg_classes=num_seg_classes)
    elif model_name == 'resnet50':
        model = resnet.resnet50(num_seg_classes=num_seg_classes)
    else: 
        raise ValueError(f"Unknown model_name: {model_name}")

    # Assign to CUDA, throw an error if no GPU
    if not no_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU available.")
        model = model.cuda()
    net_dict = model.state_dict()

    # Load the pretrained weights
    if phase == 'train' and pretrain_path:
        print(f"Loading the pretrained model from location: {pretrain_path}")                           # Print for logging
        pretrain = torch.load(pretrain_path)                                                            # Load pretrained weights
        ckpt_state = {k.replace("module.", "", 1): v for k, v in pretrain['state_dict'].items()}        # Gather pretrained weights from the checkpoint file imported, removing the module prefix
        matched_state = {k:v for k, v in ckpt_state.items() if k in net_dict.keys()}                    # Keep only items that match the scaffolding from before

        net_dict.update(matched_state)      # Update the network dictionary
        model.load_state_dict(net_dict)     # Load into our model

        not_loaded = [k for k in net_dict if k not in matched_state]                    # Find the keys that were not updated
        print(f"Loaded {len(matched_state)}/{len(net_dict)} tensors from checkpoint")   # Log how many tensors were correctly updated
        print("Unused from checkpoint:", [k for k in ckpt_state if k not in net_dict])  # See what keys from the checkpoint did not match and aren't used
        print("Randomly initialized:", not_loaded)                                      # Show which unloaded keys will train from random initialization

        new_parameters = [p for n, p in model.named_parameters() if n.split('.')[0] in new_layer_names] # Find all parameters with prefixes for the new classification head
        new_ids = {id(p) for p in new_parameters}                                                       # Create a set for fast lookup
        base_parameters = [p for p in model.parameters() if id(p) not in new_ids]                       # Find the rest of the parameters from the previous backbone

        # Print parameter structure based backbone and new classification head
        for n, p in model.named_parameters():
            tag = "[NEW] " if id(p) in new_ids else "[BASE]"
            print(tag, n, tuple(p.shape))

        parameters = {'base_parameters': base_parameters, 'new_parameters': new_parameters}
        return model, parameters

    return model, model.parameters()

