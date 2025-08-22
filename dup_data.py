import os
import shutil

# Original and target dataset paths
source_base = 'cavity_sediment_dataset'
target_base = 'cavity_sediment_dataset_augmented'
splits = ['train', 'valid', 'test']

for split in splits:
    src_img_dir = os.path.join(source_base, split, 'images')
    src_lbl_dir = os.path.join(source_base, split, 'labels')

    tgt_img_dir = os.path.join(target_base, split, 'images')
    tgt_lbl_dir = os.path.join(target_base, split, 'labels')

    os.makedirs(tgt_img_dir, exist_ok=True)
    os.makedirs(tgt_lbl_dir, exist_ok=True)

    for filename in os.listdir(src_img_dir):
        if filename.endswith(('.jpg', '.png')):
            base, ext = os.path.splitext(filename)

            # Define new names
            dup_img_name = base + "_copy" + ext
            dup_lbl_name = base + "_copy.txt"
            original_lbl_path = os.path.join(src_lbl_dir, base + ".txt")

            # Copy image with new name
            shutil.copy(os.path.join(src_img_dir, filename), os.path.join(tgt_img_dir, dup_img_name))

            # Copy label with new name (if it exists)
            if os.path.exists(original_lbl_path):
                shutil.copy(original_lbl_path, os.path.join(tgt_lbl_dir, dup_lbl_name))

print("✅ Duplicated-only dataset saved in: cavity_sediment_dataset_augmented/")
