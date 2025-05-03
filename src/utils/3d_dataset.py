class BraTSDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = glob.glob(os.path.join(data_dir, "*_flair.nii.gz"))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load flair image
        flair_path = self.image_files[idx]
        base_name = flair_path.replace("_flair.nii.gz", "")
        
        # Load other modalities
        t1_path = base_name + "_t1.nii.gz"
        t1ce_path = base_name + "_t1ce.nii.gz"
        t2_path = base_name + "_t2.nii.gz"
        seg_path = base_name + "_seg.nii.gz"
        
        # Load NIfTI files
        flair = nib.load(flair_path).get_fdata()
        t1 = nib.load(t1_path).get_fdata()
        t1ce = nib.load(t1ce_path).get_fdata()
        t2 = nib.load(t2_path).get_fdata()
        seg = nib.load(seg_path).get_fdata()
        
        # Stack modalities
        image = np.stack([flair, t1, t1ce, t2], axis=0)  # (4, D, H, W)
        
        # Preprocess
        image = self.preprocess_3d(image)
        seg = self.preprocess_3d_mask(seg)
        
        if self.transform:
            pass
            
        return torch.tensor(image, dtype=torch.float32), torch.tensor(seg, dtype=torch.float32)
    
    def preprocess_3d(self, image):
        # Normalize each modality
        for i in range(4):
            image[i] = (image[i] - image[i].mean()) / image[i].std()
        return image
    
    def preprocess_3d_mask(self, mask):
        return (mask > 0).astype(np.float32)