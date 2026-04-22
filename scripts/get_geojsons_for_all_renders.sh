echo "RUN THIS FROM REPO ROOT, PATHS ARE RELATIVE TO REPO ROOT"

for scene in cmu disaster_city pitt sample_2 sample_3 sample_4 sample_5; do
    python scripts/ges_json_to_bounds_geojson.py generated/renders/${scene}.json -o generated/renders/${scene}_loc.geojson --pad-deg 0.001
done
