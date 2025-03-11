def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler):
    model.train()
    total_loss = 0
    loss_components = {'category': 0, 'category_type': 0, 'attribute': 0, 'compatibility': 0}
    total_f1_category = 0
    total_f1_category_type = 0
    total_f1_attributes = 0
    total_accuracy_category = 0
    total_accuracy_category_type = 0
    total_accuracy_attributes = 0
    total_samples = 0

    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        images = batch['image'].to(device, non_blocking=True)
        heatmaps = batch['heatmap'].to(device, non_blocking=True)
        attributes = batch['attributes'].to(device, non_blocking=True)

        # Debug: Ensure batch is not empty
        if images.shape[0] == 0:
            raise RuntimeError(f"Empty batch received at index {batch_idx}")

        print(f"Processing batch {batch_idx} with size {images.shape[0]}")

        # Debug: Check if images were moved to the device properly
        print(f"Images tensor shape: {images.shape}, Device: {images.device}")

        # Prepare targets
        targets = {
            'category_labels': batch['category_label'].to(device, non_blocking=True),
            'category_type_labels': batch['category_type'].to(device, non_blocking=True),
            'attribute_targets': attributes,
            'compatibility_targets': torch.ones(images.size(0), 1).to(device, non_blocking=True)
        }

        # Debugging targets
        print("Targets structure:")
        for key, value in targets.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: Shape={value.shape}, Device={value.device}")
            else:
                print(f"{key}: {type(value)}")

        # Debug: Check target keys and their shapes
        print(f"Targets keys: {targets.keys()}")
        for key, value in targets.items():
            print(f" - {key}: {value.shape}, Device: {value.device}")

        # Debugging: Print shapes and min/max values before loss computation
        print("\n==== DEBUG: TRAINING BATCH ====")
        print(f"Category Labels Shape: {targets['category_labels'].shape}, Min: {targets['category_labels'].min()}, Max: {targets['category_labels'].max()}")
        print(f"Category Type Labels Shape: {targets['category_type_labels'].shape}, Min: {targets['category_type_labels'].min()}, Max: {targets['category_type_labels'].max()}")
        print(f"Attribute Targets Shape: {targets['attribute_targets'].shape}, Min: {targets['attribute_targets'].min()}, Max: {targets['attribute_targets'].max()}")
        print(f"Compatibility Targets Shape: {targets['compatibility_targets'].shape}, Min: {targets['compatibility_targets'].min()}, Max: {targets['compatibility_targets'].max()}")

        # Ensure shapes match before loss computation
        # Debugging prints before assertion
        if 'outputs' not in locals():
            raise RuntimeError("Model did not return outputs. Check model forward pass.")

        

        assert outputs['category_logits'].shape[0] == targets['category_labels'].shape[0], "Mismatch in category logits and labels"
        assert outputs['category_type_logits'].shape[0] == targets['category_type_labels'].shape[0], "Mismatch in category type logits and labels"
        assert outputs['attribute_preds'].shape == targets['attribute_targets'].shape, \
               f"Mismatch in attribute predictions {outputs['attribute_preds'].shape} and targets {targets['attribute_targets'].shape}"

        # Check for NaN or Inf values in model outputs
        for key, value in outputs.items():
            if torch.isnan(value).any() or torch.isinf(value).any():
                print(f"⚠️ Warning: {key} contains NaN or Inf values!")

        # Reset gradients
        optimizer.zero_grad()

        try:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images, heatmaps, attributes)

            # Debugging print
            if outputs is None:
                raise RuntimeError("Forward pass failed: Model returned None")

            print(f"Forward Pass Success: {outputs.keys()}" if isinstance(outputs, dict) else f"Forward Output Shape: {outputs.shape}")

            loss, loss_dict = criterion(outputs, targets)

        except Exception as e:
            raise RuntimeError(f"Error during model forward pass: {e}")

        # Compute metrics
        f1_category = compute_f1_multilabel(outputs['category_probs'], targets['category_labels'])
        accuracy_category = (outputs['category_logits'].argmax(dim=1) == targets['category_labels']).float().mean()

        f1_category_type = compute_f1_multilabel(outputs['category_type_probs'], targets['category_type_labels'])
        accuracy_category_type = (outputs['category_type_logits'].argmax(dim=1) == targets['category_type_labels']).float().mean()

        f1_attributes = compute_f1_multilabel(outputs['attribute_probs'], targets['attribute_targets'])
        accuracy_attributes = (outputs['attribute_probs'] > 0.5).float().mean()  # If thresholding

        # Backward pass with gradient clipping
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        total_loss += loss.item()
        total_f1_category += f1_category
        total_f1_category_type += f1_category_type
        total_f1_attributes += f1_attributes
        total_accuracy_category += accuracy_category
        total_accuracy_category_type += accuracy_category_type
        total_accuracy_attributes += accuracy_attributes
        total_samples += 1

        for k, v in loss_dict.items():
            if k != 'total_loss':
                loss_components[k.replace('_loss', '')] += v

    # Calculate averages
    avg_loss = total_loss / len(train_loader)
    avg_f1_category = total_f1_category / total_samples
    avg_f1_category_type = total_f1_category_type / total_samples
    avg_f1_attributes = total_f1_attributes / total_samples
    avg_accuracy_category = total_accuracy_category / total_samples
    avg_accuracy_category_type = total_accuracy_category_type / total_samples
    avg_accuracy_attributes = total_accuracy_attributes / total_samples
    avg_components = {k: v / len(train_loader) for k, v in loss_components.items()}

    # Print all metrics for transparency
    print(f"Train Loss: {avg_loss:.4f}")
    print(f"Train F1 Category: {avg_f1_category:.4f}")
    print(f"Train F1 Category Type: {avg_f1_category_type:.4f}")
    print(f"Train F1 Attributes: {avg_f1_attributes:.4f}")
    print(f"Train Accuracy Category: {avg_accuracy_category:.4f}")
    print(f"Train Accuracy Category Type: {avg_accuracy_category_type:.4f}")
    print(f"Train Accuracy Attributes: {avg_accuracy_attributes:.4f}")
    print(f"Loss Components: {avg_components}")

    return avg_loss, avg_f1_category, avg_f1_category_type, avg_f1_attributes, avg_accuracy_category, avg_accuracy_category_type, avg_accuracy_attributes, avg_components


