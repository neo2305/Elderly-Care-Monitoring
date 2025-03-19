import numpy as np
import os

def compute_pairwise_distances(sequence_coords, num_keypoints=23):
    num_frames = len(sequence_coords) // (num_keypoints * 2)
    sequence_coords = np.array(sequence_coords).reshape(num_frames, num_keypoints * 2)

    distances = []
    for i in range(num_frames - 1):
        frame1 = sequence_coords[i].reshape(num_keypoints, 2)
        frame2 = sequence_coords[i + 1].reshape(num_keypoints, 2)

        # Compute pairwise distances between keypoints of frame1 and frame2
        pairwise_dist = np.linalg.norm(frame1[:, None, :] - frame2[None, :, :], axis=-1).flatten()
        distances.append(pairwise_dist)

    return np.array(distances)

# Create output directory if needed
output_dir = "pairwise_distances_sequences"
os.makedirs(output_dir, exist_ok=True)

# Process sequences line-by-line
with open("xydat.txt", "r") as file:
    for idx, line in enumerate(file, start=1):
        line = line.strip()
        if not line:
            continue

        # Convert line to list of float coordinates
        sequence = list(map(float, line.split()))

        # Calculate number of frames in this sequence
        num_coords = len(sequence)
        if num_coords < 2 * 23 * 2:
            print(f"Skipping sequence {idx}: Not enough frames.")
            continue

        num_frames = num_coords // (23 * 2)

        # Compute distances
        distances = compute_pairwise_distances(sequence)

        # Save distances for this sequence
        output_path = os.path.join(output_dir, f"pairwise_distances_seq_{idx}.txt")
        np.savetxt(output_path, distances, fmt="%.6f")
        print(f"Saved distances for sequence {idx} to {output_path}")
