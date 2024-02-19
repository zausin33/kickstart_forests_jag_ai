
# Review

## Questions

* Why did you create steps as in the refactoring journey?
  The steps were only for illustration purposes.

* You used advanced sensai feature generator concepts. Well done! 
  Was it hard to figure out how to do the things you did?

* Did you consciously choose not to normalise the PCA features?

## Review Remarks

* `features`: `numeric_transformer` should have been a factory; otherwise
  different models will share the same instance.

* While there is one very good model, there is no comparison to simpler models
  or reduced feature sets, so we don't know what specifically contributed to 
  the success of this model (ablation study).

* Because you only considered one model, there was no need for a tracking setup,
  but it would have been good to experiment more with different types of models 
  and different parametrisations.

* `FeatureName.WAVELENGTH` is a misleading name, as it also includes the 
  sentinel features.

* `LightGBM` does not benefit from scaling/normalisation, and the
  additional `StandardScaler` at the end of the transformer chain is completely
  redundant.

* Option to sample the data in `Dataset` is not required for this use case.

* PCA feature generator unnecessarily uses `fit_transform` instead of just `fit`.