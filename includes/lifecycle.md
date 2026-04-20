<!--
Canonical Dream Trainer lifecycle diagram.

Include with: --8<-- "includes/lifecycle.md"

The sequence mirrors BaseTrainer._fit (src/dream_trainer/trainer/base.py:898)
and SetupMixin._setup_models (src/dream_trainer/trainer/mixins/setup/setup.py:108
→ mixins/setup/models.py:696-704). Update this file when the source ordering
changes; it is the single source of truth for the lifecycle diagram across
the docs.
-->

```mermaid
flowchart TD
    classDef worldStage fill:#0f1c2e,stroke:#1f3a5f,color:#cfe3ff
    classDef configStage fill:#1a1430,stroke:#3d2a6e,color:#e7d6ff
    classDef setupStage fill:#0d2818,stroke:#1f4d30,color:#cdf5d9
    classDef trainStage fill:#2b1b0a,stroke:#6e4312,color:#ffe7c2
    classDef userHook fill:#1f1f1f,stroke:#eab308,color:#fde68a,stroke-width:2px

    A["pre_launch"]:::worldStage
    B["world.launch<br/>(build device mesh)"]:::worldStage
    C["seed_everything"]:::worldStage

    D["configure_models<br/>(on meta device)"]:::userHook
    E["post_configure_models"]:::configStage

    F["apply_pipeline_parallel"]:::userHook
    G["apply_tensor_parallel"]:::userHook
    H["apply_activation_checkpointing"]:::userHook
    I["apply_compile"]:::userHook
    J["apply_fully_shard /<br/>apply_replicate"]:::userHook

    K["materialize on device"]:::setupStage
    L["init_weights"]:::userHook
    M["validate no meta tensors"]:::setupStage

    N["configure_optimizers<br/>configure_schedulers"]:::userHook
    O["configure_dataloaders<br/>configure_metrics"]:::userHook

    P["sanity validation"]:::trainStage
    Q["training epoch"]:::trainStage
    R["validation epoch"]:::trainStage
    S["post_fit"]:::trainStage

    A --> B --> C --> D --> E
    E --> F --> G --> H --> I --> J
    J --> K --> L --> M
    M --> N --> O
    O --> P --> Q
    Q <--> R
    Q --> S
```

<small>**Yellow** = hooks you implement on your trainer. **Other colors** = phases Dream Trainer owns. Parallelism hooks (PP / TP / ActCkpt / compile / FSDP or DDP) only run if the corresponding `DeviceParameters` dimension is enabled.</small>
