import hashlib
import math
from collections import Counter

def sha_bucket(category_value: str, n_buckets: int = 2**21) -> int:
    digest = hashlib.sha256(category_value.encode('utf-8')).digest()
    # Convert first 8 bytes to 64-bit integer
    val_64 = int.from_bytes(digest[:8], byteorder='big', signed=False)
    return val_64 % n_buckets

def empirical_collision_check(categories, n_buckets=2**21):
    bucket_counts = Counter()
    for cat in categories:
        b_idx = sha_bucket(cat, n_buckets=n_buckets)
        bucket_counts[b_idx] += 1
    
    # Count how many buckets have > 1 item
    collisions = sum(1 for count in bucket_counts.values() if count > 1)
    total_collisions = sum(count-1 for count in bucket_counts.values() if count > 1)
    return collisions, total_collisions

# Example usage:
categories = [f"cat_{i}" for i in range(50000)]  # or real categories
col_buckets, total_collisions = empirical_collision_check(categories, n_buckets=2**21)

print(f"Buckets with collisions: {col_buckets}")
print(f"Total collisions across all buckets: {total_collisions}")
print(f"Percent of total collisions: {total_collisions / len(categories) * 100:.2f}%")
