# Assumes torch, DataLoader, and summary are already imported in the caller's environment.

EXPECTED_CLASS_ORDER = [
    'beach', 'bus', 'cafe/restaurant', 'car', 'city_center',
    'forest_path', 'grocery_store', 'home', 'library', 'metro_station',
    'office', 'park', 'residential_area', 'train', 'tram'
]

def _all_tensors(x, torch):
    if isinstance(x, (list, tuple)):
        return all(isinstance(t, torch.Tensor) for t in x)
    return isinstance(x, torch.Tensor)

# 1) DATASET TESTER ------------------------------------------------------------
def test_dataset(LoadAudioClass, root_dir, meta_filename, audio_subdir, torch, DataLoader, *, batch_size=1):
    print("=== Dataset Tester ===")
    passed = True

    # Instantiate with the required three args
    try:
        dataset = LoadAudioClass(root_dir=root_dir, meta_filename=meta_filename, audio_subdir=audio_subdir)
        print("Dataset instantiation: OK")
    except Exception as e:
        print(f"FAILED ❌: Could not instantiate dataset: {e}")
        return False, None, None

    # Non-empty check
    ds_len = len(dataset) if hasattr(dataset, "__len__") else 0
    print(f"Number of samples: {ds_len}")
    if ds_len == 0:
        print("FAILED ❌: Dataset has zero samples. Populate `self.samples` in __init__.")
        return False, dataset, None

    # class_names checks
    class_names = getattr(dataset, "class_names", None)
    if class_names is None:
        print("FAILED ❌: dataset.class_names missing.")
        passed = False
    else:
        print(f"Possible classes ({len(class_names)}): {class_names}")

        # Type check
        if not isinstance(class_names, (list, tuple)):
            print(f"FAILED ❌: 'class_names' must be a list/tuple, got {type(class_names)}.")
            passed = False

        # Length check
        if len(class_names) != 15:
            print("FAILED ❌: class_names must contain exactly 15 classes.")
            passed = False

        # Element types
        if any(not isinstance(c, str) for c in class_names):
            print("FAILED ❌: 'class_names' must contain strings only.")
            passed = False

        # No 'placeholder'
        if any(c == "placeholder" for c in class_names):
            print("FAILED ❌: 'class_names' contains 'placeholder'—replace with real class names.")
            passed = False

        # Uniqueness
        if len(set(class_names)) != 15:
            print("FAILED ❌: 'class_names' must have 15 unique names (no duplicates).")
            passed = False

        # Sorted A–Z
        if list(class_names) != sorted(class_names):
            print("FAILED ❌: 'class_names' must be sorted alphabetically (A–Z).")
            passed = False

        # Reference list check is advisory only
        if list(class_names) == EXPECTED_CLASS_ORDER:
            print("Class order matches reference list: OK")
        else:
            print("WARNING: class order differs from the reference list (sorting must still be correct).")

    # Iterate 5 minibatches and print shapes; verify tensors & label-last convention
    try:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        print("\nIterating 5 minibatches (feature shapes):")
        got_any = False
        for i, batch_items in enumerate(loader):
            if i >= 5:
                break
            got_any = True
            print(f"  Batch {i+1}:")
            if isinstance(batch_items, (list, tuple)):
                if not _all_tensors(batch_items, torch):
                    print("  FAILED ❌: Not all returned items are torch.Tensors.")
                    passed = False
                for j, item in enumerate(batch_items):
                    print(f"    Item {j+1} shape: {tuple(item.shape)}")
            elif isinstance(batch_items, torch.Tensor):
                print("  FAILED ❌: Expected a tuple/list of tensors with label last, got a single tensor.")
                passed = False
            else:
                print(f"  FAILED ❌: Unexpected batch type: {type(batch_items)}")
                passed = False

        if not got_any:
            print("FAILED ❌: DataLoader yielded no batches (empty dataset or collate issue).")
            passed = False

    except Exception as e:
        print(f"FAILED ❌: Error iterating loader: {e}")
        passed = False
        loader = None

    print("\nDataset tests:", "PASSED ✅" if passed else "FAILED ❌")
    return passed, dataset, loader

# 2) MODEL SUMMARY PRINTER -----------------------------------------------------
def print_model_summary(model, summary, *, feature_example_shape=(1, 220500), max_params=5_000_000):
    """
    Uses the already-imported `summary` (torchsummary) from the caller.
    """
    print("\n=== Model Summary ===")
    try:
        summary(model, input_size=feature_example_shape)
    except Exception as e:
        print(f"Summary unavailable (skipping): {e}")

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", params)
    if params > max_params:
        print("FAILED ❌: Model exceeds parameter budget.")
        return False
    print("Parameter budget: OK ✅")
    return True

# 3) MODEL FORWARD TESTER ------------------------------------------------------
def test_model_forward(model, loader, torch, *, device, num_classes=15):
    print("\n=== Model Forward Tester ===")
    passed = True

    if loader is None or not hasattr(loader, "dataset") or len(loader.dataset) == 0:
        print("FAILED ❌: Loader has no data. Ensure your dataset populates `self.samples`.")
        return False

    it = iter(loader)
    try:
        batch = next(it)
    except StopIteration:
        print("FAILED ❌: Loader is empty. Cannot test model forward.")
        return False

    if torch.is_tensor(batch):
        print("FAILED ❌: Expected a tuple/list of tensors with the last item as labels.")
        return False
    if not isinstance(batch, (list, tuple)):
        print(f"FAILED ❌: Unexpected batch type: {type(batch)}")
        return False

    items = [t.to(device) if isinstance(t, torch.Tensor) else t for t in batch]

    # Split into features and labels
    label_batch = items[-1]
    features = items[:-1]

    for i, feat in enumerate(features):
        print(f"Feature {i} shape: {tuple(feat.shape)}")
    print(f"Labels shape: {tuple(label_batch.shape)}")

    # Label shape must be [1]
    if tuple(label_batch.shape) != (1,):
        print(f"FAILED ❌: Label tensor must have shape torch.Size([1]), got {tuple(label_batch.shape)}")
        passed = False

    # Optional: label dtype hint
    if not torch.is_floating_point(label_batch) and not torch.is_complex(label_batch):
        # OK (likely integer). If you want to enforce integer dtype: uncomment next line.
        # if label_batch.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        #     print(f"WARNING: Label dtype is {label_batch.dtype}; expected integer dtype.")
        pass

    # Forward pass
    model.eval()
    with torch.no_grad():
        try:
            output = model(features[0]) if len(features) == 1 else model(*features)
        except Exception as e:
            print(f"FAILED ❌: Model forward error: {e}")
            return False

    # Output checks
    if not isinstance(output, torch.Tensor):
        print("FAILED ❌: Model output is not a torch.Tensor.")
        passed = False
    else:
        print(f"Model output shape: {tuple(output.shape)}")
        expected = (1, num_classes)
        if tuple(output.shape) != expected:
            print(f"FAILED ❌: Expected {expected}, got {tuple(output.shape)}")
            passed = False

        if output.ndim == 2 and output.size(0) >= 1:
            print("Per-class logits (sample 0):")
            for i, logit in enumerate(output[0]):
                print(f"  Class {i}: {logit.item():.4f}")

    # Label value range [0, num_classes-1]
    if torch.is_tensor(label_batch):
        if not torch.all((label_batch >= 0) & (label_batch <= (num_classes - 1))):
            print(f"FAILED ❌: Labels must be integers in [0, {num_classes-1}], got values: {label_batch.tolist()}")
            passed = False

    print("Model forward check:", "OK ✅" if passed else "FAILED ❌")
    return passed