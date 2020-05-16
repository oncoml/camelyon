Workflow

I. Tile WSI's into image tiles
   - Get a list of WSI's where there tumors are detected (this can be obtained from the Camelyon17 database. We name this file "positive_slides.txt"
   - Feed "positive_files.txt" into main.py in the wsi_into_tiles directory
      -- Filter each WSI and select based on cell density
      -- For each WSI, save the tile_summary object into a list
      -- Feed each tile_summary into map_tumor module, which will map out the tumor region on each tile (if it exists). Save all tiles as images.
      -- Write list of tumor-positive tiles and tumor region coordinates into a dump file called "annotations.txt"
