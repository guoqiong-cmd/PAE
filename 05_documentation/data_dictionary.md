# Data dictionary

The modeling workflow uses the following fields.

| Field | Role |
|---|---|
| `row_ID` | Anonymous row identifier |
| `PAE` | Metadata only; not a model predictor |
| `Particle_type` | Categorical predictor |
| `MP_density` | Continuous predictor |
| `Light` | Categorical predictor |
| `Particle_size` | Continuous predictor |
| `Exposure time` | Continuous predictor |
| `PH` | Continuous predictor |
| `Temperature` | Continuous predictor |
| `Salinities` | Continuous predictor |
| `logKow` | Continuous predictor |
| `ln_RC` | Response variable |
| `condition_group_ID` | Anonymous identifier for the 122-group experimental-condition-level validation |
| `protocol_group_ID` | Retained identifier in the modeling workflow; not used by the current public validation script |
