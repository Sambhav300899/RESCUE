echo "RUN FROM REPOSITORY ROOT"

python run_full_pipeline.py --video_path generated/renders/disaster_city.mp4 --run_dir generated/full_pipeline_runs/disaster_city
python run_full_pipeline.py --video_path generated/renders/cmu.mp4 --run_dir generated/full_pipeline_runs/cmu
python run_full_pipeline.py --video_path generated/renders/pitt.mp4 --run_dir generated/full_pipeline_runs/pitt
python run_full_pipeline.py --video_path generated/renders/sample_2.mp4 --run_dir generated/full_pipeline_runs/sample_2
python run_full_pipeline.py --video_path generated/renders/sample_3.mp4 --run_dir generated/full_pipeline_runs/sample_3
python run_full_pipeline.py --video_path generated/renders/sample_4.mp4 --run_dir generated/full_pipeline_runs/sample_4
python run_full_pipeline.py --video_path generated/renders/sample_5.mp4 --run_dir generated/full_pipeline_runs/sample_5
