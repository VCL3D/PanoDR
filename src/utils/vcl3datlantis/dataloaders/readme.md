# Params required for Structured3D dataloader

	self.mask_reverse (manually) - Specifies if mask is white(False) or black(True)
	self.normalize (argument, optional) - Specifies image normalization: 'type_1' -> [-1,1], 'type_0': [0,1]
	default is 'type_1

# Usage: 

	 dataset = DatasetStructure3D(config['data_root'], 640, 480, layout="full", 
	 datum_type="normal", color = "rawlight")
	 self.train_loader = DataLoader(dataset, batch_size = (config['batch_size'], shuffle = True,
	 num_workers = 0, drop_last=True)
	 
