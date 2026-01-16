
import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Find the latest log file
log_dir = "lightning_logs"
versions = sorted(glob.glob(os.path.join(log_dir, "version_*")), key=os.path.getmtime)
latest_version = versions[-1] if versions else None

if not latest_version:
    print("No logs found.")
    exit(1)

print(f"Reading logs from: {latest_version}")
event_files = glob.glob(os.path.join(latest_version, "events.out.tfevents*"))
if not event_files:
    print("No event file found.")
    exit(1)

event_file = event_files[0]
ea = EventAccumulator(event_file)
ea.Reload()

# metrics to check
tags = ea.Tags()['scalars']
print(f"Available tags: {tags}")

if 'train_loss_step' in tags:
    events = ea.Scalars('train_loss_step')
    print(f"Found {len(events)} training steps.")
    if events:
        first = events[0]
        last = events[-1]
        duration = last.wall_time - first.wall_time
        print(f"First step time: {first.wall_time}")
        print(f"Last step time: {last.wall_time}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Initial Loss: {first.value:.4f}")
        print(f"Final Loss: {last.value:.4f}")
        
        # Approximate speed
        if duration > 0:
            speed = len(events) / duration
            print(f"Approximate speed: {speed:.2f} steps/second")
else:
    print("No train_loss_step found.")

if 'val_loss' in tags:
    val_events = ea.Scalars('val_loss')
    print(f"Validation loss: {val_events[-1].value:.4f}")
