import mediapipe as mp

print("MediaPipe attributes:")
attrs = dir(mp)
for attr in attrs:
    print(f"  {attr}")

print("\nChecking for tasks:")
if hasattr(mp, 'tasks'):
    print("  mp.tasks exists")
    print("  mp.tasks attributes:", dir(mp.tasks))
