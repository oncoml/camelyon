Workflow

1. Tile WSI's into image tiles
   File: wsi_into_tiles/main.py
   - Get a list of WSI's where there tumors are detected (this can be obtained from the Camelyon17 database. We name this file "positive_slides.txt"
   - Feed "positive_files.txt" into main.py in the wsi_into_tiles directory
      -- Filter each WSI and select based on cell density
      -- For each WSI, save the tile_summary object into a list
      -- Feed each tile_summary into map_tumor module, which will map out the tumor region on each tile (if it exists). Save all tiles as images.
      -- Write list of tumor-positive tiles and tumor region coordinates into a dump file called "annotations.txt"

2. Check tile images
   File: resnet/check_size.py
   - Check each tile image to ensure they are of proper size
   - Consider deleting or checking identified tiles

3. Prepare for training
   File: resnet/split.py; resnet/negative_tiles.py; images_mean_std.py
   - Split the tile images into different cross-validation and tumor sets 
   - Save negative tiles into each train and val nontumor directories
   - Calculate images mean

4. Train resnet50
   File: train.py
   - Train a resnet50 model
