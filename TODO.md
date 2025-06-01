## TODO List

### Code Refactoring
- [ ] Implement a unified logger:
      - Redirect all `print` statements to both terminal and `outputs/run.log`
      - Create a `Logger` class and replace `print` with `logger.info()`
- [x] Restructure configuration handling:
      - Move all settings into the `Config` class, except the current experiment file (e.g. `python main.py --config your_config`)
- [ ] Fix augmentation arguments to be fully handled via config (Gaia)
- [x] Migrate argument parsing into config:
      - Optimizer settings
      - Training hyperparameters
      - Map constants 
      - ⚠️ Evaluate whether all args should move to config (except for config file name and folder)

### Experiments

#### General
- [x] Finalize dataset preparation (Filippo)
- [ ] Write comprehensive test coverage (Gaia) for:
      - Dataloader
      - Model
      - Experiment pipeline 
- [x] Clean up experimental code (Filippo)



#### Specific Experiments
- [x] Re-run random baseline experiments (Filippo — max 5 mins)
- [x] Run experiments on `objects_unseen` split (Gaia)
- [ ] Run experiments on `scene_unseen` split (Gaia)
- [ ] Add positional encodings to map? Inside MHA?

- [ ] 🥂 Pop a bottle if it works

### Future Work
- [ ] Extend dataset to multi-target per episode (e.g. “find one of Julia’s jackets”)
- [ ] Add support for “shared” objects (e.g. “the jacket belongs to both Julia and Mark”)
- [ ] Negation? "the couch belongs to filippo and NOT to tommaso", "belongs to everyone except to filippo"