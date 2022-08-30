# IRFNet-and-CUGAN-in-simple-cli

Simple python script to run [Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) and [IFRNet](https://github.com/ltkong218/IFRNet) to upscale and interpolate video. Can be used separately.

# Usage:

```
usage: main.py [-h] [-i INPUT] [-o OUTPUT] [--input_type {video,images}] [--base_fps BASE_FPS]
                 [--images_ext IMAGES_EXT] [--upscaler_model {pro-conservative,pro-denoise3x,pro-no-denoise}]
                 [-m {upscale,interpolate,upscale-interpolate,interpolate-upscale}] [-u UPSCALE]
                 [--upscale_tile UPSCALE_TILE] [--IFRNet_model {IFRNet,IFRNetL}] [-f FPS_MULTIP]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to input video file or dir with images (use --input_type to choose type of input)
  -o OUTPUT, --output OUTPUT
  --input_type {video,images}
  --base_fps BASE_FPS   Used only if input is dir with images, by default it is 23.97602397602398 (equialent to
                        24000/1001)
  --images_ext IMAGES_EXT
  --upscaler_model {pro-conservative,pro-denoise3x,pro-no-denoise}
                        Model type, right file will be choised by upscale factor
  -m {upscale,interpolate,upscale-interpolate,interpolate-upscale}, --mode {upscale,interpolate,upscale-interpolate,interpolate-upscale}
  -u UPSCALE, --upscale UPSCALE
  --upscale_tile UPSCALE_TILE
  --IFRNet_model {IFRNet,IFRNetL}
  -f FPS_MULTIP, --fps_multip FPS_MULTIP
  ```
  
  ```--mode upscale-interpolate``` requires much more VRAM than ``` --mode interpolate-upscale```
  
