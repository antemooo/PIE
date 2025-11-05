python3 pie/pipeline.py \
    --data-dir ../PPMI \
    --output-dir ./output/motor_assessments_NP1RTOT_final \
    --target-column motor_assessments_NP1RTOT \
    --leakage-features-path config/leakage_features.txt \
    --fs-method fdr \
    --fs-param 0.05 \
    --n-models 5 \
    --tune \
    --budget 60.0
