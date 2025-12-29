#!/usr/bin/env python3
"""
SRDE Training Monitor

Monitor a running training job via heartbeat file.
Can also send notifications when training state changes.
"""
import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timedelta


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor SRDE training")
    parser.add_argument("--output_dir", type=str, default="./srde_checkpoints")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--timeout", type=int, default=300, help="Consider stalled after N seconds")
    parser.add_argument("--webhook", type=str, default=None, help="Discord/Slack webhook URL")
    return parser.parse_args()


def send_notification(webhook_url: str, message: str):
    """Send a notification via webhook."""
    if not webhook_url:
        return
    
    import urllib.request
    import urllib.error
    
    data = json.dumps({"content": f"🤖 SRDE Monitor: {message}"}).encode('utf-8')
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    
    try:
        urllib.request.urlopen(req, timeout=10)
    except urllib.error.URLError as e:
        print(f"[WARN] Failed to send notification: {e}")


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    
    print("="*60)
    print("SRDE Training Monitor")
    print("="*60)
    print(f"Monitoring: {output_dir}")
    print(f"Check interval: {args.interval}s")
    print(f"Stall timeout: {args.timeout}s")
    print("="*60)
    
    last_step = -1
    last_alert_time = None
    training_started = False
    
    while True:
        try:
            # Check for completion
            complete_file = output_dir / "TRAINING_COMPLETE"
            if complete_file.exists():
                data = json.loads(complete_file.read_text())
                print(f"\n✅ TRAINING COMPLETE!")
                print(f"   Final step: {data.get('final_step', 'unknown')}")
                print(f"   Best loss: {data.get('best_loss', 'unknown')}")
                print(f"   Checkpoint: {data.get('final_checkpoint', 'unknown')}")
                send_notification(args.webhook, f"✅ Training complete! Final step: {data.get('final_step')}")
                break
            
            # Check heartbeat
            heartbeat_file = output_dir / "heartbeat.json"
            if heartbeat_file.exists():
                heartbeat = json.loads(heartbeat_file.read_text())
                
                step = heartbeat.get("step", 0)
                loss = heartbeat.get("loss", 0)
                phase = heartbeat.get("phase", "unknown")
                timestamp_str = heartbeat.get("timestamp", "")
                
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    age = (datetime.now() - timestamp).total_seconds()
                except:
                    age = 0
                
                # Detect new training
                if not training_started:
                    training_started = True
                    print(f"\n🚀 Training detected!")
                    send_notification(args.webhook, "🚀 Training started!")
                
                # Detect progress
                if step != last_step:
                    eta = "calculating..."
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step {step:,} | Loss: {loss:.4f} | Phase: {phase}")
                    last_step = step
                    last_alert_time = None  # Reset alert
                
                # Detect stall
                if age > args.timeout:
                    if last_alert_time is None or (datetime.now() - last_alert_time).total_seconds() > 600:
                        print(f"\n⚠️  WARNING: Training may be stalled! Last heartbeat: {format_duration(age)} ago")
                        send_notification(args.webhook, f"⚠️ Training stalled? No heartbeat for {format_duration(age)}")
                        last_alert_time = datetime.now()
            
            else:
                # No heartbeat yet
                started_file = output_dir / "TRAINING_STARTED"
                if started_file.exists():
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training started, waiting for first heartbeat...")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for training to start...")
            
            time.sleep(args.interval)
            
        except KeyboardInterrupt:
            print("\n\nMonitor stopped.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
